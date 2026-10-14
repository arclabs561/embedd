#![cfg(any(feature = "openai", feature = "async-openai"))]

use embedd::EmbedMode;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::thread;
use std::time::{Duration, Instant};

// Synthetic successful embedding responses: data/embedding follow the adapter's
// wire types; object/index/model/usage follow the OpenAI embeddings response.
// Only cardinality varies. This does not test an index or permutation policy.
fn response_server(rows: usize) -> (String, thread::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    listener.set_nonblocking(true).unwrap();
    let handle = thread::spawn(move || {
        let deadline = Instant::now() + Duration::from_secs(10);
        let mut stream = loop {
            match listener.accept() {
                Ok((stream, _)) => break stream,
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    assert!(Instant::now() < deadline, "local request not received");
                    thread::sleep(Duration::from_millis(5));
                }
                Err(e) => panic!("local accept failed: {e}"),
            }
        };
        stream
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        stream
            .set_write_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        let mut request = Vec::new();
        let mut buf = [0; 1024];
        let (body_start, content_length) = loop {
            let n = stream.read(&mut buf).unwrap();
            assert!(n > 0, "request ended before headers");
            request.extend_from_slice(&buf[..n]);
            assert!(request.len() < 16384, "synthetic request exceeds bound");
            if let Some(end) = request.windows(4).position(|w| w == b"\r\n\r\n") {
                let headers = std::str::from_utf8(&request[..end]).unwrap();
                assert!(headers.starts_with("POST /v1/embeddings HTTP/1.1\r\n"));
                let length = headers
                    .lines()
                    .find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        name.eq_ignore_ascii_case("content-length")
                            .then(|| value.trim().parse::<usize>().unwrap())
                    })
                    .expect("content length");
                assert!(length < 4096);
                break (end + 4, length);
            }
        };
        while request.len() < body_start + content_length {
            let n = stream.read(&mut buf).unwrap();
            assert!(n > 0, "request ended before body");
            request.extend_from_slice(&buf[..n]);
        }
        let input: serde_json::Value =
            serde_json::from_slice(&request[body_start..body_start + content_length]).unwrap();
        assert_eq!(
            input["input"],
            serde_json::json!(["first synthetic input", "second synthetic input"])
        );
        assert_eq!(input["model"], "local-synthetic-model");
        let vectors = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let data: Vec<_> = vectors[..rows].iter().enumerate().map(|(index, vector)|
            serde_json::json!({"object": "embedding", "index": index, "embedding": vector})
        ).collect();
        let body = serde_json::json!({"object": "list", "data": data,
            "model": "local-synthetic-model", "usage": {"prompt_tokens": 4, "total_tokens": 4}})
        .to_string();
        write!(stream, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", body.len(), body).unwrap();
    });
    (url, handle)
}

fn assert_count_contract(result: anyhow::Result<Vec<Vec<f32>>>, rows: usize) {
    if rows == 2 {
        assert_eq!(result.unwrap(), vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    } else {
        assert!(
            result.is_err(),
            "two inputs with {rows} response rows must be rejected; got {result:?}"
        );
    }
}

#[cfg(feature = "openai")]
fn sync_case(rows: usize) {
    use embedd::TextEmbedder;
    let (url, server) = response_server(rows);
    let adapter =
        embedd::openai::OpenAiEmbedder::new("synthetic-not-a-key", "local-synthetic-model")
            .with_base_url(url);
    let result = adapter.embed_texts(
        &[
            "first synthetic input".into(),
            "second synthetic input".into(),
        ],
        EmbedMode::Document,
    );
    server.join().unwrap();
    assert_count_contract(result, rows);
}

#[cfg(feature = "async-openai")]
async fn async_case(rows: usize) {
    use embedd::AsyncTextEmbedder;
    let (url, server) = response_server(rows);
    let client = reqwest::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(10))
        .build()
        .unwrap();
    let adapter = embedd::async_openai::AsyncOpenAiEmbedder::new(
        "synthetic-not-a-key",
        "local-synthetic-model",
    )
    .with_base_url(url)
    .with_client(client);
    let result = adapter
        .embed_texts(
            &[
                "first synthetic input".into(),
                "second synthetic input".into(),
            ],
            EmbedMode::Document,
        )
        .await;
    server.join().unwrap();
    assert_count_contract(result, rows);
}

#[cfg(feature = "openai")]
#[test]
fn sync_valid_distinct_control() {
    sync_case(2);
}
#[cfg(feature = "openai")]
#[test]
fn sync_rejects_short_success() {
    sync_case(1);
}
#[cfg(feature = "openai")]
#[test]
fn sync_rejects_extra_success() {
    sync_case(3);
}
#[cfg(feature = "openai")]
#[test]
fn sync_rejects_empty_success() {
    sync_case(0);
}
#[cfg(feature = "async-openai")]
#[tokio::test]
async fn async_valid_distinct_control() {
    async_case(2).await;
}
#[cfg(feature = "async-openai")]
#[tokio::test]
async fn async_rejects_short_success() {
    async_case(1).await;
}
#[cfg(feature = "async-openai")]
#[tokio::test]
async fn async_rejects_extra_success() {
    async_case(3).await;
}
#[cfg(feature = "async-openai")]
#[tokio::test]
async fn async_rejects_empty_success() {
    async_case(0).await;
}
