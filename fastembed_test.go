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
	fmt.Println(result[0][0])
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
	fmt.Println(result[0][0])
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
	fmt.Println(result[0][0])
	if len(result) != len(input) {
		t.Errorf("Expected result length %v, got %v", len(input), len(result))
	}
}
