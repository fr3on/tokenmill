use std::path::PathBuf;
use hf_hub::api::sync::Api;
use hf_hub::{Repo, RepoType};
use anyhow::{Context, Result};

pub fn download(repo_id: &str, filename: &str, revision: Option<&str>, repo_type: RepoType) -> Result<PathBuf> {
    let api = Api::new().context("Failed to create HF API client")?;

    let repo = if let Some(rev) = revision {
        api.repo(Repo::with_revision(repo_id.to_string(), repo_type, rev.to_string()))
    } else {
        api.repo(Repo::new(repo_id.to_string(), repo_type))
    };

    let path = repo.get(filename)?;
    Ok(path)
}

/// List all files inside a HF repo. Returns (rfilename, size_in_bytes).
pub fn list_repo(repo_id: &str, repo_type: RepoType) -> Result<Vec<(String, Option<u64>)>> {
    let api = Api::new().context("Failed to create HF API client")?;
    let repo = api.repo(Repo::new(repo_id.to_string(), repo_type));
    let info = repo.info().context("Failed to fetch repo info")?;
    let files = info
        .siblings
        .into_iter()
        .map(|s| (s.rfilename, None::<u64>))
        .collect();
    Ok(files)
}

pub fn resolve_hf_uri(uri: &str) -> Result<PathBuf> {
    if !uri.starts_with("hf://") {
        anyhow::bail!("Not a Hugging Face URI");
    }

    let parts: Vec<&str> = uri[5..].split('/').collect();
    if parts.len() < 2 {
        anyhow::bail!("Invalid hf:// URI. Expected hf://repo_id/filename");
    }

    let repo_id = parts[0..parts.len() - 1].join("/");
    let filename = parts[parts.len() - 1];

    download(&repo_id, filename, None, RepoType::Dataset)
}
