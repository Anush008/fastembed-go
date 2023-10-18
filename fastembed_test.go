package fastembed

import (
	"math"
	"testing"
)

func TestCanonicalValues(T *testing.T) {
	canonicalValues := map[EmbeddingModel]([]float32){
		AllMiniLML6V2: []float32{0.02591, 0.00573, 0.01147, 0.03796, -0.02328},
		BGESmallEN:    []float32{-0.02313, -0.02552, 0.017357, -0.06393, -0.00061},
		BGEBaseEN:     []float32{0.01140, 0.03722, 0.02941, 0.01230, 0.03451},
		BGEBaseENV15:  []float32{0.01129394, 0.05493144, 0.02615099, 0.00328772, 0.02996045},
		BGESmallENV15: []float32{0.01522374, -0.02271799, 0.00860278, -0.07424029, 0.00386434},
		BGESmallZH:    []float32{-0.01023294, 0.07634465, 0.0691722, -0.04458365, -0.03160762},
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

		epsilon := float64(1e-4)
		for i, v := range expected {
			if math.Abs(float64(result[0][i]-v)) > float64(epsilon) {
				T.Errorf("Element %d mismatch for %s: expected %.6f, got %.6f", i, model, v, result[0][i])
			}
		}
	}

}
