package fastembed

import (
	"fmt"
	"os"
	"reflect"
	"testing"
)

func TestNewFlagEmbedding(t *testing.T) {
	// Test with default options
	options := &InitOptions{
		OnnxPath: os.Getenv("ONNX_PATH"),
	}
	_, err := NewFlagEmbedding(options)
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}
}

func TestEmbed(t *testing.T) {
	// Test with a single input
	options := &InitOptions{
		OnnxPath: os.Getenv("ONNX_PATH"),
		Model:    AllMiniLML6V2,
	}
	fe, err := NewFlagEmbedding(options)
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
	options := &InitOptions{
		OnnxPath: os.Getenv("ONNX_PATH"),
	}
	fe, err := NewFlagEmbedding(options)
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
	options := &InitOptions{
		OnnxPath: os.Getenv("ONNX_PATH"),
	}
	fe, err := NewFlagEmbedding(options)
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
