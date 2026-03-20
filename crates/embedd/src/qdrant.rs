//! Bridge between embedd embedders and Qdrant vector database.
//!
//! Provides [`embed_and_upsert`] and [`embed_and_search`] for the common
//! "embed text, then store/query in Qdrant" workflow.
//!
//! Requires the `qdrant` feature.

use anyhow::Result;
use qdrant_client::qdrant::{
    PointStruct, SearchPointsBuilder, SearchResponse, UpsertPointsBuilder,
};
use qdrant_client::Qdrant;

use crate::{EmbedMode, TextEmbedder};

/// Embed texts and upsert them as points into a Qdrant collection.
///
/// Each text gets a sequential point ID starting from `start_id`.
/// The embedding vector is stored as the default (unnamed) vector.
/// Payload includes `"text"` with the original string.
pub async fn embed_and_upsert(
    client: &Qdrant,
    collection: &str,
    embedder: &dyn TextEmbedder,
    texts: &[String],
    start_id: u64,
    mode: EmbedMode,
) -> Result<()> {
    if texts.is_empty() {
        return Ok(());
    }

    let vectors = embedder.embed_texts(texts, mode)?;

    let points: Vec<PointStruct> = texts
        .iter()
        .zip(vectors)
        .enumerate()
        .map(|(i, (text, vec))| {
            let id = start_id + i as u64;
            PointStruct::new(id, vec, [("text".to_string(), text.clone().into())])
        })
        .collect();

    let request = UpsertPointsBuilder::new(collection, points);
    client.upsert_points(request).await?;

    Ok(())
}

/// Embed a query and search for similar points in a Qdrant collection.
pub async fn embed_and_search(
    client: &Qdrant,
    collection: &str,
    embedder: &dyn TextEmbedder,
    query: &str,
    limit: u64,
) -> Result<SearchResponse> {
    let query_vec = embedder.embed_text(query, EmbedMode::Query)?;

    let request = SearchPointsBuilder::new(collection, query_vec, limit).with_payload(true);

    let response = client.search_points(request).await?;
    Ok(response)
}
