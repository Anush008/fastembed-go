<div align="center">
 <h1 style="display: inline-block; vertical-align: middle;">
    <a href="https://crates.io/crates/fastembed">FastEmbed-go</a>
    <img src="https://github.com/Anush008/fastembed-rs/assets/46051506/4bd3cefe-12da-48b9-8cc2-7489145c9cb5" style="display: inline-block; vertical-align: middle; width: auto; height: 100px;">
 </h1>
 <h3>Go implementation of <a href="https://github.com/qdrant/fastembed" target="_blank">@Qdrant/fastembed</a></h3>
  <a href="https://pkg.go.dev/github.com/anush008/fastembed-go"><img src="https://pkg.go.dev/badge/github.com/anush008/fastembed-go.svg" alt="Go Reference"></a>
  <a href="https://github.com/Anush008/fastembed-go/blob/master/LICENSE"><img src="https://img.shields.io/badge/license-mit-blue.svg" alt="MIT Licensed"></a>
  <a href="https://github.com/Anush008/fastembed-go/actions/workflows/release.yml"><img src="https://github.com/Anush008/fastembed-go/actions/workflows/release.yml/badge.svg?branch=main" alt="Semantic release"></a>
</div>

## 🍕 Features

- Supports batch embeddings with parallelism using go-routines.
- Uses [@sugarme/tokenizer](https://github.com/sugarme/tokenizer) for fast tokenization.
- Optimized embedding models.

The default embedding supports "query" and "passage" prefixes for the input text. The default model is Flag Embedding, which is top of the [MTEB](https://huggingface.co/spaces/mteb/leaderboard) leaderboard.

## 🔍 Not looking for Go?

- Python 🐍: [fastembed](https://github.com/qdrant/fastembed)
- Rust 🦀: [fastembed-rs](https://github.com/Anush008/fastembed-rs)
- JavaScript 🌐: [fastembed-js](https://github.com/Anush008/fastembed-js)
  
## 🤖 Models

- [**BAAI/bge-base-en**](https://huggingface.co/BAAI/bge-base-en)
- [**BAAI/bge-base-en-v1.5**](https://huggingface.co/BAAI/bge-base-en-v1.5)
- [**BAAI/bge-small-en**](https://huggingface.co/BAAI/bge-small-en)
- [**BAAI/bge-small-en-v1.5**](https://huggingface.co/BAAI/bge-small-en-v1.5) - Default
- [**BAAI/bge-base-zh-v1.5**](https://huggingface.co/BAAI/bge-base-zh-v1.5)
- [**sentence-transformers/all-MiniLM-L6-v2**](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

## 🚀 Installation

Run the following Go CLI command in your project directory:

```bash
go get -u github.com/Anush008/fastembed-go
```

## 📖 Usage

```go
import "github.com/anush008/fastembed-go"

// With default options
model, err := fastembed.NewFlagEmbedding(nil)
if err != nil {
 panic(err)
}
defer model.Destroy()

// With custom options
options := fastembed.InitOptions{
 Model:     fastembed.BGEBaseEN,
 CacheDir:  "model_cache",
 MaxLength: 200,
}

model, err = fastembed.NewFlagEmbedding(&options)
if err != nil {
 panic(err)
}
defer model.Destroy()

documents := []string{
 "passage: Hello, World!",
 "query: Hello, World!",
 "passage: This is an example passage.",
 // You can leave out the prefix but it's recommended
 "fastembed-go is licensed under MIT",
}

// Generate embeddings with a batch-size of 25, defaults to 256
embeddings, err := model.Embed(documents, 25)  //  -> Embeddings length: 4
if err != nil {
 panic(err)
}
```

### Supports passage and query embeddings for more accurate results

```go
// Generate embeddings for the passages
// The texts are prefixed with "passage" for better results
// The batch size is set to 1 for demonstration purposes
passages := []string{
 "This is the first passage. It contains provides more context for retrieval.",
 "Here's the second passage, which is longer than the first one. It includes additional information.",
 "And this is the third passage, the longest of all. It contains several sentences and is meant for more extensive testing.",
}

embeddings, err := model.PassageEmbed(passages, 1)  //  -> Embeddings length: 3
if err != nil {
 panic(err)
}

// Generate embeddings for the query
// The text is prefixed with "query" for better retrieval
query := "What is the answer to this generic question?";

embeddings, err := model.QueryEmbed(query)
if err != nil {
 panic(err)
}
```

## ℹ︎ Notice:

The Onnx runtime path is automatically loaded on most environments. However, if you encounter
```sh
panic: Platform-specific initialization failed: Error loading ONNX shared library
```
Set the `ONNX_PATH` env to your Onnx installation.
For eg, on MacOS:
```sh
export ONNX_PATH="/path/to/onnx/lib/libonnxruntime.dylib"
```
On Linux:
```sh
export ONNX_PATH="/path/to/onnx/lib/libonnxruntime.so"
```
You can find the Onnx runtime releases [here](https://github.com/microsoft/onnxruntime/releases).

## 🚒 Under the hood

### Why fast?

It's important we justify the "fast" in FastEmbed. FastEmbed is fast because:

1. Quantized model weights
2. ONNX Runtime which allows for inference on CPU, GPU, and other dedicated runtimes

### Why light?

1. No hidden dependencies via Huggingface Transformers

### Why accurate?

1. Better than OpenAI Ada-002
2. Top of the Embedding leaderboards e.g. [MTEB](https://huggingface.co/spaces/mteb/leaderboard)

## 📄 LICENSE

MIT © [2023](https://github.com/Anush008/fastembed-go/blob/main/LICENSE)
