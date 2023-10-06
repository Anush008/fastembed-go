package main

import (
	fastembed "fastembed/pkg"
	"fmt"
)

func main() {
	model, err := fastembed.NewFlagEmbedding(&fastembed.InitOptions{
		Model: fastembed.BGESmallEN,
	})

	if err != nil {
		panic(err)
	}

	documents := []string{
		"Is the world doing okay?",
	}
	embeddings, err := model.Embed(documents, 0)
	if err != nil {
		panic(err)
	}

	fmt.Println(embeddings[0][0])
}
