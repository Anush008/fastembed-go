package fastembed

import (
	"archive/tar"
	"compress/gzip"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"sync"

	"github.com/schollz/progressbar/v3"
	"github.com/sugarme/tokenizer"
	"github.com/sugarme/tokenizer/pretrained"
	ort "github.com/yalue/onnxruntime_go"
)

// Enum-type representing the available embedding models
type EmbeddingModel string

const (
	AllMiniLML6V2 EmbeddingModel = "fast-all-MiniLM-L6-v2"
	BGEBaseEN     EmbeddingModel = "fast-bge-base-en"
	BGESmallEN    EmbeddingModel = "fast-bge-small-en"

// A model with type "Unigram" is not yet supported by the tokenizer
// Ref: https://github.com/sugarme/tokenizer/blob/448e79b1ed65947b8c6343bf9aa39e78364f45c8/pretrained/model.go#L152
// MLE5Large     EmbeddingModel = "fast-multilingual-e5-large"
)

// Struct to interface with a FastEmbed model
type FlagEmbedding struct {
	tokenizer *tokenizer.Tokenizer
	model     EmbeddingModel
	maxLength int
	modelPath string
}

// Options to initialize a FastEmbed model
// Model: The model to use for embedding
// ExecutionProviders: The execution providers to use for onnxruntime
// MaxLength: The maximum length of the input sequence
// CacheDir: The directory to cache the model files
// ShowDownloadProgress: Whether to show the download progress bar
// NOTE:
// We use a pointer for "ShowDownloadProgress" so that we can distinguish between the user
// not setting this flag and the user setting it to false. We want the default value to be true.
// As Go assigns a default(empty) value of "false" to bools, we can't distinguish
// if the user set it to false or not set at all.
// A pointer to bool will be nil if not set explicitly
type InitOptions struct {
	Model                EmbeddingModel
	ExecutionProviders   []string
	MaxLength            int
	CacheDir             string
	ShowDownloadProgress *bool
}

// Struct to represent FastEmbed model information
type ModelInfo struct {
	Model       EmbeddingModel
	Dim         int
	Description string
}

// Function to initialize a FastEmbed model
func NewFlagEmbedding(options *InitOptions) (*FlagEmbedding, error) {
	if options.CacheDir == "" {
		options.CacheDir = "local_cache"
	}

	if options.Model == "" {
		options.Model = BGESmallEN
	}

	if options.MaxLength == 0 {
		options.MaxLength = 512
	}

	if options.ShowDownloadProgress == nil {
		showDownloadProgress := true
		options.ShowDownloadProgress = &showDownloadProgress
	}

	if onnxPath := os.Getenv("ONNX_PATH"); onnxPath != "" {
		ort.SetSharedLibraryPath(onnxPath)
	}

	if !ort.IsInitialized() {
		err := ort.InitializeEnvironment()
		if err != nil {
			return nil, err
		}
	}

	modelPath, err := retrieveModel(options.Model, options.CacheDir, *options.ShowDownloadProgress)
	if err != nil {
		return nil, err
	}

	tknzer, err := loadTokenizer(modelPath, options.MaxLength)
	if err != nil {
		return nil, err
	}
	return &FlagEmbedding{
		tokenizer: tknzer,
		model:     options.Model,
		maxLength: options.MaxLength,
		modelPath: modelPath,
	}, nil

}

// Function to cleanup the internal onnxruntime environment when it is no longer needed.
func (f *FlagEmbedding) Destroy() {
	ort.DestroyEnvironment()
}

// Private function to embed a batch of input strings
func (f *FlagEmbedding) onnxEmbed(input []string) ([]([]float32), error) {

	inputs := make([]tokenizer.EncodeInput, len(input))
	for index, v := range input {
		sequence := tokenizer.NewInputSequence(v)
		inputs[index] = tokenizer.NewSingleEncodeInput(sequence)
	}

	encodings, err := f.tokenizer.EncodeBatch(inputs, true)
	if err != nil {
		return nil, err
	}

	inputIdsFlat, inputMaskFlat, inputTypeIdsFlat := make([]int64, 0), make([]int64, 0), make([]int64, 0)
	for _, encoding := range encodings {
		inputIds, inputMask, inputTypeIds := encodingToInt32(encoding.GetIds(), encoding.GetAttentionMask(), encoding.GetTypeIds())
		inputIdsFlat = append(inputIdsFlat, inputIds...)
		inputMaskFlat = append(inputMaskFlat, inputMask...)
		inputTypeIdsFlat = append(inputTypeIdsFlat, inputTypeIds...)
	}

	inputShape := ort.NewShape(int64(len(inputs)), int64(f.maxLength))

	inputTensorID, err := ort.NewTensor(inputShape, inputIdsFlat)
	if err != nil {
		return nil, err
	}
	defer inputTensorID.Destroy()

	inputTensorMask, err := ort.NewTensor(inputShape, inputMaskFlat)

	if err != nil {
		return nil, err
	}
	defer inputTensorMask.Destroy()

	inputTensorType, err := ort.NewTensor(inputShape, inputTypeIdsFlat)

	if err != nil {
		return nil, err
	}
	defer inputTensorType.Destroy()

	modelInfo, err := getModelInfo(f.model)
	if err != nil {
		return nil, err
	}

	outputShape := ort.NewShape(int64(len(inputs)), int64(f.maxLength), int64(modelInfo.Dim))
	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		return nil, err
	}
	defer outputTensor.Destroy()

	session, err := ort.NewAdvancedSession(filepath.Join(f.modelPath, "model_optimized.onnx"), []string{
		"input_ids", "attention_mask", "token_type_ids",
	}, []string{
		"last_hidden_state",
	}, []ort.ArbitraryTensor{
		inputTensorID, inputTensorMask, inputTensorType,
	}, []ort.ArbitraryTensor{outputTensor},
		nil)

	if err != nil {
		return nil, err
	}

	defer session.Destroy()

	err = session.Run()
	if err != nil {
		return nil, err
	}

	return getEmbeddings(outputTensor.GetData(), outputTensor.GetShape()), nil
}

// Function to embed a batch of input strings
// The batchSize parameter controls the number of inputs to embed in a single batch
// The batches are processed in parallel
// Returns the first error encountered if any
// Default batch size is 512
func (f *FlagEmbedding) Embed(input []string, batchSize int) ([]([]float32), error) {
	if batchSize <= 0 {
		batchSize = 512
	}
	embeddings := make([]([]float32), len(input))
	var wg sync.WaitGroup
	errorCh := make(chan error, len(input))
	//var resultsMutex sync.Mutex

	for i := 0; i < len(input); i += batchSize {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			end := i + batchSize
			if end > len(input) {
				end = len(input)
			}
			batchOut, err := f.onnxEmbed(input[i:end])
			if err != nil {
				errorCh <- err
			}
			// resultsMutex.Lock()
			// defer resultsMutex.Unlock()
			//Removed the mutex as the slice positions being accessed are unique for each goroutine and there is no overlap
			copy(embeddings[i:end], batchOut)

		}(i)
	}
	wg.Wait()
	close(errorCh)

	// We can aggregate the errors if we ever need to
	if len(errorCh) > 0 {
		return nil, <-errorCh
	}
	return embeddings, nil
}

// Function to embed a single input string prefixed with "query: "
// Recommended for generating query embeddings for semantic search
func (f *FlagEmbedding) QueryEmbed(input string) ([]float32, error) {
	query := "query: " + input
	data, err := f.onnxEmbed([]string{query})
	if err != nil {
		return nil, err
	}
	return data[0], nil
}

// Function to embed string prefixed with "passage: "
func (f *FlagEmbedding) PassageEmbed(input []string, batchSize int) ([]([]float32), error) {
	processedInput := make([]string, len(input))
	for i, v := range input {
		processedInput[i] = "passage: " + v
	}
	return f.Embed(processedInput, batchSize)
}

// Function to list the supported FastEmbed models
func ListSupportedModels() []ModelInfo {
	return []ModelInfo{
		{
			Model:       AllMiniLML6V2,
			Dim:         384,
			Description: "Sentence Transformer model, MiniLM-L6-v2",
		},
		{
			Model:       BGEBaseEN,
			Dim:         768,
			Description: "Base English model",
		},
		{
			Model:       BGESmallEN,
			Dim:         384,
			Description: "Fast and Default English model",
		},
		// {
		// 	Model:       MLE5Large,
		// 	Dim:         1024,
		// 	Description: "Multilingual model, e5-large. Recommend using this model for non-English languages",
		// },
	}
}

// TODO: Configure the from model config files
func loadTokenizer(modelPath string, maxLength int) (*tokenizer.Tokenizer, error) {
	tknzer, err := pretrained.FromFile(filepath.Join(modelPath, "tokenizer.json"))

	if err != nil {
		return nil, err
	}

	maxLen := maxLength

	tknzer.WithTruncation(&tokenizer.TruncationParams{
		MaxLength: maxLen,
		Strategy:  tokenizer.LongestFirst,
		Stride:    0,
	})

	padToken := "[PAD]"
	paddingStrategy := tokenizer.NewPaddingStrategy(tokenizer.WithFixed(maxLen))

	paddingParams := tokenizer.PaddingParams{
		Strategy:  *paddingStrategy,
		Direction: tokenizer.Right,
		PadId:     0,
		PadToken:  padToken,
	}
	tknzer.WithPadding(&paddingParams)

	return tknzer, nil
}

// Private function to get model information from the model name
func getModelInfo(model EmbeddingModel) (ModelInfo, error) {
	for _, m := range ListSupportedModels() {
		if m.Model == model {
			return m, nil
		}
	}
	return ModelInfo{}, fmt.Errorf("model %s not found", model)
}

// Private function to retrieve the model from the cache or download it
// Returns the path to the model
func retrieveModel(model EmbeddingModel, cacheDir string, showDownloadProgress bool) (string, error) {
	if _, err := os.Stat(filepath.Join(cacheDir, string(model))); !errors.Is(err, fs.ErrNotExist) {
		return filepath.Join(cacheDir, string(model)), nil
	}
	return downloadFromGcs(model, cacheDir, showDownloadProgress)
}

// Private function to download the model from Google Cloud Storage
func downloadFromGcs(model EmbeddingModel, cacheDir string, showDownloadProgress bool) (string, error) {
	// The MLE5Large model URL doesn't follow the same naming convention as the other models
	// So, we tranform "fast-multilingual-e5-large" -> "intfloat-multilingual-e5-large" in the download URL
	// The model directory name in the GCS storage is "fast-multilingual-e5-large", like the others
	// modelName := model
	// if model == MLE5Large {
	// 	modelName = "intfloat" + model[strings.Index(string(model), "-"):]
	// }

	downloadURL := fmt.Sprintf("https://storage.googleapis.com/qdrant-fastembed/%s.tar.gz", model)

	response, err := http.Get(downloadURL)
	if err != nil {
		return "", err
	}
	defer response.Body.Close()

	if response.StatusCode < 200 || response.StatusCode > 299 {
		return "", fmt.Errorf("model download failed: %s", response.Status)
	}

	if showDownloadProgress {
		bar := progressbar.DefaultBytes(
			response.ContentLength,
			"Downloading "+string(model),
		)
		reader := progressbar.NewReader(response.Body, bar)
		err = untar(&reader, cacheDir)
	} else {
		fmt.Printf("Downloading %s...", model)
		err = untar(response.Body, cacheDir)
	}

	if err != nil {
		return "", err
	}

	return filepath.Join(cacheDir, string(model)), nil
}

// Private function to untar the downloaded model from a .tar.gz file
func untar(tarball io.Reader, target string) error {
	archive, err := gzip.NewReader(tarball)
	if err != nil {
		return err
	}
	defer archive.Close()

	tarReader := tar.NewReader(archive)

	for {
		header, err := tarReader.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}

		path := filepath.Join(target, header.Name)
		switch header.Typeflag {
		case tar.TypeDir:
			if err := os.MkdirAll(path, 0755); err != nil {
				return err
			}
		case tar.TypeReg:
			if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
				return err
			}

			file, err := os.Create(path)
			if err != nil {
				return err
			}
			defer file.Close()
			if _, err := io.Copy(file, tarReader); err != nil {
				return err
			}
		}
	}
	return nil
}

// Private function to normalize a vector
// Based on https://github.com/qdrant/fastembed/blob/ca6f9d629ad14da1dfd094c846976b0c964b32cf/fastembed/embedding.py#L16
func normalize(v []float32) []float32 {
	norm := float32(0.0)
	for _, val := range v {
		norm += val * val
	}
	norm = float32(math.Sqrt(float64(norm)))
	epsilon := float32(1e-12)

	normalized := make([]float32, len(v))
	for i, val := range v {
		normalized[i] = (val / norm) + epsilon
	}

	return normalized
}

// Private function to return the normalized embeddings from a flattened array with the given dimensions
func getEmbeddings(data []float32, dimensions []int64) []([]float32) {
	x, y, z := dimensions[0], dimensions[1], dimensions[2]
	embeddings := make([][]float32, x)
	var i int64
	for i = 0; i < x; i++ {
		startIndex := i * y * z
		endIndex := startIndex + z
		embeddings[i] = normalize(data[startIndex:endIndex])
	}
	return embeddings
}

// Private function to convert multiple int32 slices to int64 slices as required by the onnxruntime API
// With a linear time complexity
func encodingToInt32(inputA, inputB, inputC []int) (outputA, outputB, outputC []int64) {
	if len(inputA) != len(inputB) || len(inputB) != len(inputC) {
		panic("input lengths do not match")
	}
	outputA = make([]int64, len(inputA))
	outputB = make([]int64, len(inputB))
	outputC = make([]int64, len(inputC))
	for i := range inputA {
		outputA[i] = int64(inputA[i])
		outputB[i] = int64(inputB[i])
		outputC[i] = int64(inputC[i])
	}
	return
}
