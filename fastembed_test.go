package fastembed

import (
	"fmt"
	"reflect"
	"testing"
)

func TestNewFlagEmbedding(t *testing.T) {
	// Test with default options
	_, err := NewFlagEmbedding(&InitOptions{})
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}
}

func TestEmbed(t *testing.T) {
	// Test with a single input
	fe, err := NewFlagEmbedding(&InitOptions{})
	defer fe.Destroy()
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}
	input := []string{"Is the world doing okay?"}
	result, err := fe.Embed(input, 1)
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}
	fmt.Println(result[0][0])
	if len(result) != len(input) {
		t.Errorf("Expected result length %v, got %v", len(input), len(result))
	}
}

func TestQueryEmbed(t *testing.T) {
	// Test with a single input
	fe, err := NewFlagEmbedding(&InitOptions{})
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}
	input := "Hello, world!"
	result, err := fe.QueryEmbed(input)
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}
	if len(result) == 0 {
		t.Errorf("Expected non-empty result")
	}
}

func TestPassageEmbed(t *testing.T) {
	// Test with a single input
	fe, err := NewFlagEmbedding(&InitOptions{})
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}
	input := []string{"Hello, world!"}
	result, err := fe.PassageEmbed(input, 1)
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}
	if len(result) != len(input) {
		t.Errorf("Expected result length %v, got %v", len(input), len(result))
	}
}

func TestEncodingToInt32(t *testing.T) {
	inputA := []int{1, 2, 3}
	inputB := []int{4, 5, 6}
	inputC := []int{7, 8, 9}
	outputA, outputB, outputC := encodingToInt32(inputA, inputB, inputC)
	expectedA := []int64{1, 2, 3}
	expectedB := []int64{4, 5, 6}
	expectedC := []int64{7, 8, 9}
	if !reflect.DeepEqual(outputA, expectedA) {
		t.Errorf("Expected %v, got %v", expectedA, outputA)
	}
	if !reflect.DeepEqual(outputB, expectedB) {
		t.Errorf("Expected %v, got %v", expectedB, outputB)
	}
	if !reflect.DeepEqual(outputC, expectedC) {
		t.Errorf("Expected %v, got %v", expectedC, outputC)
	}
}

// // Define the canonical vector values as a map
// var canonicalVectorValues = map[string][]float64{
// 	"BAAI/bge-small-en":                      {-0.0232, -0.0255, 0.0174, -0.0639, -0.0006},
// 	"BAAI/bge-base-en":                       {0.0115, 0.0372, 0.0295, 0.0121, 0.0346},
// 	"sentence-transformers/all-MiniLM-L6-v2": {0.0259, 0.0058, 0.0114, 0.0380, -0.0233},
// 	"intfloat/multilingual-e5-large":         {0.0098, 0.0045, 0.0066, -0.0354, 0.0070},
// }

// // Define the test for default embedding
// func TestDefaultEmbedding(t *testing.T) {
// 	for _, modelDesc := range Embedding.ListSupportedModels() {
// 		dim := modelDesc["dim"]
// 		model := DefaultEmbedding(modelDesc["model"])

// 		docs := []string{"hello world", "flag embedding"}
// 		embeddings := model.Embed(docs)
// 		if len(embeddings) != 2 || len(embeddings[0]) != dim {
// 			t.Errorf("Expected embeddings shape (2, %v), got (%v, %v)", dim, len(embeddings), len(embeddings[0]))
// 		}

// 		canonicalVector := canonicalVectorValues[modelDesc["model"]]
// 		for i, val := range embeddings[0][:len(canonicalVector)] {
// 			if math.Abs(val-canonicalVector[i]) > 1e-3 {
// 				t.Errorf("Expected %v, got %v", canonicalVector[i], val)
// 			}
// 		}
// 	}
// }

// // Define the test for batch embedding
// func TestBatchEmbedding(t *testing.T) {
// 	model := DefaultEmbedding()

// 	docs := make([]string, 200)
// 	for i := range docs {
// 		if i%2 == 0 {
// 			docs[i] = "hello world"
// 		} else {
// 			docs[i] = "flag embedding"
// 		}
// 	}
// 	embeddings := model.Embed(docs, 10)
// 	if len(embeddings) != 200 || len(embeddings[0]) != 384 {
// 		t.Errorf("Expected embeddings shape (200, 384), got (%v, %v)", len(embeddings), len(embeddings[0]))
// 	}
// }
