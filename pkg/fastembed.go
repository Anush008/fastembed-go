package fastembed

import (
	"archive/tar"
	"compress/gzip"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"net/http"
	"os"
	"path/filepath"
	"sync"

	"github.com/schollz/progressbar/v3"
	"github.com/sugarme/tokenizer"
	"github.com/sugarme/tokenizer/pretrained"
	ort "github.com/yalue/onnxruntime_go"
)

func encodingToInt32(inputA, inputB, inputC []int) (outputA, outputB, outputC []int64) {
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

// Enum-type representing the available embedding models
type EmbeddingModel string

const (
	AllMiniLML6V2 EmbeddingModel = "fast-all-MiniLM-L6-v2"
	BGEBaseEN     EmbeddingModel = "fast-bge-base-en"
	BGESmallEN    EmbeddingModel = "fast-bge-small-en"
	// MLE5Large     EmbeddingModel = "intfloat-multilingual-e5-large"
)

type FlagEmbedding struct {
	tokenizer *tokenizer.Tokenizer
	model     EmbeddingModel
	maxLength int
}

type InitOptions struct {
	Model              EmbeddingModel
	ExecutionProviders []string
	MaxLength          int
	CacheDir           string
	// Whether to show download progress when downloading the model
	// NOTE:
	// We use a pointer here so that we can distinguish between the user
	// not setting this flag and the user setting it to false.
	// As Go assigns a default(empty) value of "false" to bools, we can't distinguish
	// if the user set it to false or not set at all.
	// A pointer to bool will be nil if not set explicitly
	ShowDownloadProgress *bool
}

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

	ort.SetSharedLibraryPath("/opt/homebrew/Cellar/onnxruntime/1.16.0/lib/libonnxruntime.1.16.0.dylib")
	err := ort.InitializeEnvironment()
	if err != nil {
		return nil, err
	}

	modelPath, err := retrieveModel(options.Model, options.CacheDir, *options.ShowDownloadProgress)
	if err != nil {
		return nil, err
	}

	tknzer, err := pretrained.FromFile(filepath.Join(modelPath, "tokenizer.json"))

	if err != nil {
		return nil, err
	}

	maxLen := options.MaxLength

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
	return &FlagEmbedding{
		tokenizer: tknzer,
		model:     options.Model,
		maxLength: maxLen,
	}, nil

}

// Call this function to cleanup the internal onnxruntime environment when it is no longer needed.
func (f *FlagEmbedding) Destroy() {
	ort.DestroyEnvironment()
}

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

	outputShape := ort.NewShape(int64(len(inputs)), int64(f.maxLength), 384)
	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		return nil, err
	}
	defer outputTensor.Destroy()

	session, err := ort.NewAdvancedSession("/Users/anush/Desktop/fastembed-go/local_cache/fast-bge-small-en/model_optimized.onnx", []string{
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

func getEmbeddings(data []float32, dimensions []int64) []([]float32) {
	x, y, z := dimensions[0], dimensions[1], dimensions[2]
	embeddings := make([][]float32, x)
	var i int64
	for i = 0; i < x; i++ {
		startIndex := i * y * z
		endIndex := startIndex + z
		embeddings[i] = data[startIndex:endIndex]
	}
	return embeddings
}

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

func retrieveModel(model EmbeddingModel, cacheDir string, showDownloadProgress bool) (string, error) {
	if _, err := os.Stat(filepath.Join(cacheDir, string(model))); !errors.Is(err, fs.ErrNotExist) {
		return filepath.Join(cacheDir, string(model)), nil
	}
	return downloadFromGcs(model, cacheDir, showDownloadProgress)
}

func downloadFromGcs(model EmbeddingModel, cacheDir string, showDownloadProgress bool) (string, error) {
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
		err = Untar(&reader, cacheDir)
	} else {
		err = Untar(response.Body, cacheDir)
	}

	if err != nil {
		return "", err
	}
	return filepath.Join(cacheDir, string(model)), nil
}

func Untar(tarball io.Reader, target string) error {
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