/*

Package regexpset is an extension to Go standard library `regexp`. It contains a set
of regular expression patterns and provides methods for matching on input
string.

## Basic example

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
            t.Log(err)
        }

        setmatches := set.Matches("foobar")

        fmt.Println(setmatches.Matches())

        // [0 1 2 3 4 6]
    }

*/
package regexpset
