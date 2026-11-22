use gguf::Gguf;

use crate::{Error, Result};

/// Model hyperparameters read from an `inkling2_loop` GGUF. Field meanings: docs/reference/ARCHITECTURE.md.
#[derive(Debug, Clone)]
pub struct Hparams {
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub n_vocab_padded: usize,
    pub n_vocab: usize,
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub n_ff_exp: usize,
    pub rms_eps: f32,
    pub qk_clip: f32,
    pub rel_clip: f32,
    pub n_rel: usize,
    pub rel_extent: usize,
    pub n_swa: usize,
    pub is_swa: Vec<bool>,
    pub n_conv: usize,
    pub attn_res_block: usize,
    pub log_scale_floor: f32,
    pub log_scale_alpha: f32,
    pub situ_beta: f32,
    pub situ_linear_beta: f32,
    pub expert_weights_scale: f32,
    pub final_softcap: f32,
    pub loop_start: usize,
    pub loop_end: usize,
    pub loop_passes: usize,
    pub loop_gate_rank: usize,
}

impl Hparams {
    pub fn from_gguf(g: &Gguf) -> Result<Self> {
        let arch = g.get_str("general.architecture")?;
        if arch != "inkling2_loop" {
            return Err(Error::Unsupported(format!(
                "architecture {arch}, expected inkling2_loop"
            )));
        }
        let k = |s: &str| format!("inkling2_loop.{s}");
        let u = |s: &str| -> Result<usize> { Ok(g.get_u64(&k(s))? as usize) };
        let f = |s: &str| -> Result<f32> { Ok(g.get_f32(&k(s))?) };

        let n_layer = u("block_count")?;
        let heads = g.get_u64_per_layer(&k("attention.head_count"), n_layer)?;
        let heads_kv = g.get_u64_per_layer(&k("attention.head_count_kv"), n_layer)?;
        let is_swa = g
            .get_array(&k("attention.sliding_window_pattern"))?
            .iter()
            .map(|v| matches!(v, gguf::Value::Bool(true)))
            .collect::<Vec<_>>();

        // The engine assumes every layer shares one attention shape; the GGUF allows per-layer
        // arrays, so reject files where that does not hold instead of computing garbage.
        if heads.iter().any(|&h| h != heads[0]) || heads_kv.iter().any(|&h| h != heads_kv[0]) {
            return Err(Error::Unsupported("per-layer head counts".into()));
        }
        let head_dim = u("attention.key_length")?;
        if u("attention.key_length_swa")? != head_dim
            || u("attention.value_length")? != head_dim
            || u("attention.value_length_swa")? != head_dim
        {
            return Err(Error::Unsupported(
                "different SWA / value head sizes".into(),
            ));
        }

        let hp = Hparams {
            n_embd: u("embedding_length")?,
            n_layer,
            n_head: heads[0] as usize,
            n_head_kv: heads_kv[0] as usize,
            head_dim,
            n_vocab_padded: u("vocab_size")?,
            n_vocab: u("vocab_size_unpadded")?,
            n_expert: u("expert_count")?,
            n_expert_used: u("expert_used_count")?,
            n_ff_exp: u("expert_feed_forward_length")?,
            rms_eps: f("attention.layer_norm_rms_epsilon")?,
            qk_clip: f("attention.qk_clip")?,
            rel_clip: f("attention.relative_clip")?,
            n_rel: u("attention.relative_length")?,
            rel_extent: u("attention.relative_extent")?,
            n_swa: u("attention.sliding_window")?,
            is_swa,
            n_conv: u("shortconv.l_cache")?,
            attn_res_block: u("attn_res.block_size")?,
            log_scale_floor: u("attention.log_scale_floor")? as f32,
            log_scale_alpha: f("attention.log_scale_alpha")?,
            situ_beta: f("activation.situ_beta")?,
            situ_linear_beta: f("activation.situ_linear_beta")?,
            expert_weights_scale: f("expert_weights_scale")?,
            final_softcap: f("final_logit_softcapping")?,
            loop_start: u("loop.start")?,
            loop_end: u("loop.end")?,
            loop_passes: u("loop.count")?,
            loop_gate_rank: u("loop.gate_rank")?,
        };

        if g.get_bool(&k("loop.varying")).unwrap_or(false) {
            return Err(Error::Unsupported("variable-loop models".into()));
        }
        let b = hp.attn_res_block;
        if hp.is_swa.len() != n_layer
            || !hp.n_head.is_multiple_of(hp.n_head_kv)
            || hp.loop_start == 0
            || hp.loop_start >= hp.loop_end
            || hp.loop_end > n_layer
            || !hp.loop_start.is_multiple_of(b)
            || !hp.loop_end.is_multiple_of(b)
            || hp.loop_passes < 2
            || hp.n_vocab > hp.n_vocab_padded
        {
            return Err(Error::Unsupported(format!(
                "inconsistent hyperparameters: {hp:?}"
            )));
        }
        Ok(hp)
    }

    pub fn n_loop(&self) -> usize {
        self.loop_end - self.loop_start
    }

    /// Cache slots: one per layer application (physical layers + extra passes over the core).
    pub fn n_slots(&self) -> usize {
        self.n_layer + (self.loop_passes - 1) * self.n_loop()
    }

    /// Cache slot of layer `il` in loop pass `pass` (0-based). Matches llama.cpp's slot numbering,
    /// which is also the layer suffix on its debug tensor names.
    pub fn slot(&self, il: usize, pass: usize) -> usize {
        if pass == 0 || il < self.loop_start || il >= self.loop_end {
            il
        } else {
            self.n_layer + (pass - 1) * self.n_loop() + (il - self.loop_start)
        }
    }

    pub fn kv_dim(&self) -> usize {
        self.n_head_kv * self.head_dim
    }

    pub fn q_dim(&self) -> usize {
        self.n_head * self.head_dim
    }

    /// `1 + alpha*ln(max(1, (pos+1)/floor))`, applied to q and the relative states on full layers.
    pub fn tau(&self, pos: usize) -> f32 {
        1.0 + self.log_scale_alpha * ((pos as f32 + 1.0) / self.log_scale_floor).max(1.0).ln()
    }
}
