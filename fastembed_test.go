package fastembed_test

import (
	"math"
	"testing"

	fastembed "github.com/anush008/fastembed-go"
)

func TestCanonicalValues(t *testing.T) {
	canonicalValues := map[fastembed.EmbeddingModel]([]float32){
		fastembed.AllMiniLML6V2: []float32{0.02591, 0.00573, 0.01147, 0.03796, -0.02328},
		fastembed.BGESmallEN:    []float32{-0.02313, -0.02552, 0.017357, -0.06393, -0.00061},
		fastembed.BGEBaseEN:     []float32{0.01140, 0.03722, 0.02941, 0.01230, 0.03451},
		fastembed.BGEBaseENV15:  []float32{0.01129394, 0.05493144, 0.02615099, 0.00328772, 0.02996045},
		fastembed.BGESmallENV15: []float32{0.01522374, -0.02271799, 0.00860278, -0.07424029, 0.00386434},
		fastembed.BGESmallZH:    []float32{-0.01023294, 0.07634465, 0.0691722, -0.04458365, -0.03160762},
	}

	for model, expected := range canonicalValues {
		fe, err := fastembed.NewFlagEmbedding(&fastembed.InitOptions{
			Model: model,
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

		epsilon := float64(1e-4)
		for i, v := range expected {
			if math.Abs(float64(result[0][i]-v)) > epsilon {
				t.Errorf("Element %d mismatch for %s: expected %.6f, got %.6f", i, model, v, result[0][i])
			}
		}
	}
}
