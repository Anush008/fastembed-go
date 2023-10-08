package fastembed

import (
	"fmt"
	"testing"
)

// TODO: Added canonical tests for all models
func TestEmbedBGESmallEN(t *testing.T) {
	// Test with a single input
	fe, err := NewFlagEmbedding(&InitOptions{
		Model: BGESmallEN,
	})
	defer fe.Destroy()
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}
	input := []string{"Is the world doing okay?"}
	result, err := fe.Embed(input, 1)
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}

	fmt.Printf("result: %v\n", result[0][0:10])
	if len(result) != len(input) {
		t.Errorf("Expected result length %v, got %v", len(input), len(result))
	}
}

func TestEmbedBGEBaseEN(t *testing.T) {
	// Test with a single input
	fe, err := NewFlagEmbedding(&InitOptions{
		Model: BGEBaseEN,
	})
	defer fe.Destroy()
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}
	input := []string{"Is the world doing okay?"}
	result, err := fe.Embed(input, 1)
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}

	fmt.Printf("result: %v\n", result[0][0:10])
	if len(result) != len(input) {
		t.Errorf("Expected result length %v, got %v", len(input), len(result))
	}
}

func TestEmbedAllMiniLML6V2(t *testing.T) {
	// Test with a single input
	fe, err := NewFlagEmbedding(&InitOptions{
		Model: AllMiniLML6V2,
	})
	defer fe.Destroy()
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}
	input := []string{"Is the world doing okay?"}
	result, err := fe.Embed(input, 1)
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}

	fmt.Printf("result: %v\n", result[0][0:10])
	if len(result) != len(input) {
		t.Errorf("Expected result length %v, got %v", len(input), len(result))
	}
}

// A model type "Unigram" is not yet supported by the tokenizer
// Ref: https://github.com/sugarme/tokenizer/blob/448e79b1ed65947b8c6343bf9aa39e78364f45c8/pretrained/model.go#L152
// func TestEmbedMLE5Large(t *testing.T) {
// 	// Test with a single input
// 	show := false
// 	fe, err := NewFlagEmbedding(&InitOptions{
// 		Model:                MLE5Large,
// 		ShowDownloadProgress: &show,
// 	})
// 	defer fe.Destroy()
// 	if err != nil {
// 		t.Fatalf("Expected no error, got %v", err)
// 	}
// 	input := []string{"Is the world doing okay?"}
// 	result, err := fe.Embed(input, 1)
// 	if err != nil {
// 		t.Fatalf("Expected no error, got %v", err)
// 	}

// 	if len(result) != len(input) {
// 		t.Errorf("Expected result length %v, got %v", len(input), len(result))
// 	}
// }
