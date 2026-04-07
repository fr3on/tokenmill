use candle_core::{Device, Tensor, DType};
use candle_nn::{VarBuilder};
use candle_transformers::models::qwen2::{Config as Qwen2Config, ModelForCausalLM as Qwen2Model};
use candle_transformers::models::bert::{Config as BertConfig, BertModel};
use tokenizers::Tokenizer;
use anyhow::Result;
use std::path::Path;

pub struct Scorer {
    model: Qwen2Model,
    tokenizer: Tokenizer,
    device: Device,
}

impl Scorer {
    pub fn load_qwen2(model_dir: &Path) -> Result<Self> {
        let device = Device::Cpu; 
        
        let config_path = model_dir.join("config.json");
        let tokenizer_path = model_dir.join("tokenizer.json");
        let weights_path = model_dir.join("model.safetensors");
        
        let config: Qwen2Config = serde_json::from_reader(std::fs::File::open(config_path)?)?;
        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(anyhow::Error::msg)?;
        
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)?
        };
        
        let model = Qwen2Model::new(&config, vb)?;
        
        Ok(Self { model, tokenizer, device })
    }

    pub fn compute_perplexity(&mut self, text: &str) -> Result<f32> {
        let tokens = self.tokenizer.encode(text, true).map_err(anyhow::Error::msg)?;
        let token_ids = tokens.get_ids();
        let seq_len = token_ids.len();
        
        if seq_len < 2 {
            return Ok(0.0);
        }
        
        let input = Tensor::new(token_ids, &self.device)?.unsqueeze(0)?;
        
        // Clear KV cache before each line to ensure independence
        self.model.clear_kv_cache();
        
        let logits = self.model.forward(&input, 0)?;
        let labels = input.narrow(1, seq_len - 1, 1)?.flatten(0, 1)?;
        let loss = candle_nn::loss::cross_entropy(&logits, &labels)?;
        let ppl = loss.to_scalar::<f32>()?.exp();
        
        Ok(ppl)
    }
}

pub struct Embedder {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
}

impl Embedder {
    pub fn load_bert(model_dir: &Path) -> Result<Self> {
        let device = Device::Cpu;
        
        let config_path = model_dir.join("config.json");
        let tokenizer_path = model_dir.join("tokenizer.json");
        let weights_path = model_dir.join("model.safetensors");
        
        let config: BertConfig = serde_json::from_reader(std::fs::File::open(config_path)?)?;
        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(anyhow::Error::msg)?;
        
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)?
        };
        
        let model = BertModel::load(vb, &config)?;
        
        Ok(Self { model, tokenizer, device })
    }

    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let tokens = self.tokenizer.encode(text, true).map_err(anyhow::Error::msg)?;
        let token_ids = tokens.get_ids();
        let token_type_ids = tokens.get_type_ids();
        
        let token_ids = Tensor::new(token_ids, &self.device)?.unsqueeze(0)?;
        let token_type_ids = Tensor::new(token_type_ids, &self.device)?.unsqueeze(0)?;
        
        // Create a basic attention mask (all ones)
        let attention_mask = token_ids.ones_like()?;
        
        let embeddings = self.model.forward(&token_ids, &token_type_ids, Some(&attention_mask))?;
        
        let (_n_batch, n_tokens, _hidden_size) = embeddings.dims3()?;
        let embeddings = (embeddings.sum(1)? / (n_tokens as f64))?;
        let embeddings = embeddings.get(0)?;
        
        Ok(embeddings.to_vec1::<f32>()?)
    }
}
