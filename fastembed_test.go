package fastembed

import (
	"math"
	"testing"
)

func TestEmbedBGEBaseEN(t *testing.T) {
	// Test with a single input
	fe, err := NewFlagEmbedding(&InitOptions{
		Model: BGEBaseEN,
	})
	defer fe.Destroy()
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}
	input := []string{"hello world"}
	result, err := fe.Embed(input, 1)
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}

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
	input := []string{"hello world"}
	result, err := fe.Embed(input, 1)
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}

	if len(result) != len(input) {
		t.Errorf("Expected result length %v, got %v", len(input), len(result))
	}
}

func TestEmbedBGESmallEN(t *testing.T) {
	// Test with a single input
	fe, err := NewFlagEmbedding(&InitOptions{
		Model: BGESmallEN,
	})
	defer fe.Destroy()
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}
	input := []string{"hello world"}
	result, err := fe.Embed(input, 1)
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}

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
// 	input := []string{"hello world"}
// 	result, err := fe.Embed(input, 1)
// 	if err != nil {
// 		t.Fatalf("Expected no error, got %v", err)
// 	}

// 	if len(result) != len(input) {
// 		t.Errorf("Expected result length %v, got %v", len(input), len(result))
// 	}
// }

func TestCanonicalValues(T *testing.T) {
	canonicalValues := map[EmbeddingModel]([]float32){
		AllMiniLML6V2: []float32{0.02591, 0.00573, 0.01147, 0.03796, -0.02328, -0.05493, 0.014040, -0.01079, -0.02440, -0.01822},
		BGESmallEN:    []float32{-0.02313, -0.02552, 0.017357, -0.06393, -0.00061, 0.02212, -0.01472, 0.03925, 0.03444, 0.00459},
		BGEBaseEN:     []float32{0.01140, 0.03722, 0.02941, 0.01230, 0.03451, 0.00876, 0.02356, 0.05414, -0.02945, -0.05472},
	}

	for model, expected := range canonicalValues {
		fe, err := NewFlagEmbedding(&InitOptions{
			Model: model,
		})
		defer fe.Destroy()
		if err != nil {
			T.Fatalf("Expected no error, got %v", err)
		}
		input := []string{"hello world"}
		result, err := fe.Embed(input, 1)
		if err != nil {
			T.Fatalf("Expected no error, got %v", err)
		}

		if len(result) != len(input) {
			T.Errorf("Expected result length %v, got %v", len(input), len(result))
		}

		epsilon := float64(1e-5)
		for i, v := range expected {
			if math.Abs(float64(result[0][i]-v)) > float64(epsilon) {
				T.Errorf("Element %d mismatch: expected %.6f, got %.6f", i, v, result[0][i])
			}
		}
	}

}
