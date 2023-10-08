package regexpset

// regexpset is an extension of standard library `regexp`.
// It contains a set of regular expressions and provides methods to do matching on this set.
// It is equivalent to Rust [RegexSet](https://docs.rs/regex/1.3.9/regex/struct.RegexSet.html)

import (
	"fmt"
	"log"
	"regexp"
)

// RegexpSet :
// ===========

// RegexpSet contains a set of regular expression set for matching.
type RegexpSet struct {
	patterns []string
}

// NewRegexpSet creates a new RegexpSet from a slice of patterns
func NewRegexpSet(patterns []string) (retVal *RegexpSet, err error) {
	for _, pattern := range patterns {
		_, err := regexp.Compile(pattern)
		if err != nil {
			err = fmt.Errorf("Invalid regular pattern: '%v'.\n", pattern)
			return nil, err
		}
	}

	return &RegexpSet{patterns}, nil
}

// TODO: NewRegexpSetFromIter
// Creates a new RegexpSet from an iterator of patterns (string).
// This takes an iterator of string. If any of the string is not
// valid regular expression, an error will be returned.

// IsMatch returns true if and only if one of the regular expressions in
// this set matches the given input text.
func (rs RegexpSet) IsMatch(s string) bool {
	for _, pattern := range rs.patterns {
		r := regexp.MustCompile(pattern)
		if r.MatchString(s) {
			return true
		}
	}
	return false
}

// IsMatchAt returns the same as `IsMatch`, but starts the
// search at the given offset.
//
// NOTE. the significance of the starting point is that it
// takes the surrounding context into consideration. E.g.
// `\A` anchor can only match when `start == 0`.
func (rs RegexpSet) IsMatchAt(s string, start int) bool {
	runes := []rune(s) // TODO. should we use []rune or []byte???
	if start < 0 || start >= len(runes) {
		log.Fatalf("'start' index is out of bound. \n")
	}
	substring := string(runes[start:])
	return rs.IsMatch(substring)
}

// Matches returns indexes of patterns those match input string
func (rs RegexpSet) Matches(s string) (retVal SetMatches) {
	var matches []bool = make([]bool, len(rs.patterns))
	var matchedAny bool = false
	for i, pattern := range rs.patterns {
		r := regexp.MustCompile(pattern)
		matches[i] = r.MatchString(s)
	}

	for _, m := range matches {
		if m {
			matchedAny = true
			break
		}
	}

	return SetMatches{
		matchedAny: matchedAny,
		matches:    matches,
	}
}

//ReadMatchesAt returns the same as `Matches`, but start the search
// at a given offset.
//
// NOTE. the significance of the starting point is that it takes
// the surrounding context into consideration. For example, the
// `/A` anchor can only match when `start == 0`.
//
// This method returns `true` if and only if at least one member of
// `matches` is `true` after executing the set against input string.
func (rs RegexpSet) ReadMatchesAt(s string, start int) bool {
	runes := []rune(s) // TODO. should we use []rune or []byte???
	if start < 0 || start >= len(runes) {
		log.Fatalf("'start' index is out of bound. \n")
	}

	substring := string(runes[start:])
	for _, pattern := range rs.patterns {
		r := regexp.MustCompile(pattern)
		if r.MatchString(substring) {
			return true
		}
	}

	return false
}

// Len returns total number of regular expression in this set.
func (rs RegexpSet) Len() int {
	return len(rs.patterns)
}

// Patterns returns the patterns this set will match on.
func (rs RegexpSet) Patterns() []string {
	return rs.patterns
}

// SetMatches :
// ============

// SetMatches is a set of matches returned by a RegexpSet
type SetMatches struct {
	matchedAny bool
	matches    []bool
}

// MatchedAny returns whether this set contains any matches.
func (sm SetMatches) MatchedAny() bool {
	return sm.matchedAny
}

// Matched returns whether the regular expression pattern at given index matched
// The index for a regular expression pattern is determined by its insersion
// order upon the initial construction of a `RegexpSet`, starting from 0.
// It will be panic if patternIdx is out of bound.
func (sm SetMatches) Matched(patternIdx int) bool {
	if patternIdx < 0 || patternIdx >= len(sm.matches) {
		log.Fatalf("Invalid input pattern index: %v. It is out of bound.\n", patternIdx)
	}

	return sm.matches[patternIdx]
}

// Len returns the total number of regular expressions in the set that created
// these matches.
func (sm SetMatches) Len() int {
	return len(sm.matches)
}

// Matches returns indexes of regular expression that matched.
func (sm SetMatches) Matches() (retVal []int) {
	for i, m := range sm.matches {
		if m {
			retVal = append(retVal, i)
		}
	}
	return retVal
}

// Iter returns an iterator over the indexes in the regular expression that
// matched.
// This will always produce matches in ascending order of index, where the index
// corresponds to the index of the regular expressions that matched with respect
// to its position when initial building the set.
func (sm SetMatches) Iter() SetMatchesIter {

	return SetMatchesIter{
		matches: sm.matches,
		nextIdx: 0,
	}
}

// SetMatchesIter is an SetMatches iterator
type SetMatchesIter struct {
	matches []bool
	nextIdx int
}

// Next implement iterator for SetMatchesIter
// It returns INDEX of regular expression pattern that
// matched. It will return a INDEX value = -1 if not matched.
func (smi *SetMatchesIter) Next() (retVal int, ok bool) {
	if smi.nextIdx >= len(smi.matches) {
		return -1, false
	}

	if smi.matches[smi.nextIdx] {
		retVal = smi.nextIdx
	} else {
		retVal = -1
	}
	smi.nextIdx++

	return retVal, true
}
