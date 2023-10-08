# regexpset [![License](https://img.shields.io/:license-apache-blue.svg)](https://opensource.org/licenses/Apache-2.0)[![Go.Dev reference](https://img.shields.io/badge/go.dev-reference-007d9c?logo=go&logoColor=white&style=flat-square)](https://pkg.go.dev/github.com/sugarme/regexpset?tab=doc)[![TravisCI](https://api.travis-ci.org/sugarme/regexpset.svg?branch=master)](https://travis-ci.org/sugarme/regexpset)[![Go Report Card](https://goreportcard.com/badge/github.com/sugarme/regexpset)](https://goreportcard.com/report/github.com/sugarme/regexpset)

## What is it?

`regexpset` is an extension to Go standard library `regexp`. It contains a set
of regular expression patterns and provides methods for matching on input
string. 

## Basic example

```Go
    
    import "github.com/sugarme/regexpset"

    func main(){
        var patterns []string = []string{
            `\w+`,
            `\d+`,
            `\pL+`,
            `foo`,
            `bar`,
            `barfoo`,
            `foobar`,
        }

        set, err := regexpset.NewRegexpSet(patterns)
        if err != nil {
            log.Fatal(err)
        }

        setmatches := set.Matches("foobar")

        fmt.Println(setmatches.Matches())

        // [0 1 2 3 4 6]
    }
   
```

## Documentation 

See [Go.Dev Reference](https://pkg.go.dev/github.com/sugarme/regexpset?tab=doc)


## License: Apache 2.0



