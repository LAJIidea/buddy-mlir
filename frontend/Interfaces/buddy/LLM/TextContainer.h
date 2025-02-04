//===- TextContainer.h ----------------------------------------------------===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// Text container descriptor.
//
// TODO-LOW: Add a generic tokenizer.
//
//===----------------------------------------------------------------------===//

#ifndef FRONTEND_INTERFACES_BUDDY_LLM_TEXTCONTAINER
#define FRONTEND_INTERFACES_BUDDY_LLM_TEXTCONTAINER

#include "buddy/Core/Container.h"
#include <fstream>
#include <iostream>
#include <unordered_map>

namespace buddy {

// Text container.
// - T represents the type of the elements.
// - N represents the number of dimensions.
template <typename T, size_t N> class Text : public MemRef<T, N> {
public:
  // Text Constructor with string.
  // This constructor initializes a Text object with the provided string.
  // The provided string is stored internally for tokenization and processing.
  Text(const std::string &str);
  // Bert Tokenizer
  // Tokenize an input string based on vocabulary and container size.
  // This function initializes the necessary memory references and sets up the
  // structure for storing tokenized data.
  // The vocabulary file is read to build a token-to-id map for efficient token
  // processing.
  // The input string is iterated character by character, and tokens are
  // extracted based on whitespace and punctuation.
  // Tokens are processed using the `processToken` function and stored in the
  // allocated memory.
  // Special tokens (e.g., [CLS] and [SEP]) are added at the beginning and end
  // of the tokenized sequence.
  void tokenizeBert(const std::string &vocab, size_t length, bool lower = true,
                    bool affix = false);
  // LLAMA Tokenizer
  // This function initializes the necessary memory references and sets up the
  // structure for storing tokenized data.
  // Different from the base tokenizer, this function implements the tokenize
  // by scoring the substring and select the best matching token.
  // Read the string at once, and replace all whitespace with a special
  // mark — thick underline.
  void tokenizeLlama(const std::string &vocab, size_t length);

  // Revert the ids into tokens.
  // This function initializes the conversion from Text memref to a string.
  // Tokens are identified by ids and thick underlines are replaced with
  // whitespaces.
  std::string revert(Text<T, 2> input);

  // Get sequence length
  size_t getTokenCnt() { return this->tokenCnt; }
  // Set sequence length
  void setTokenCnt(size_t cnt) { this->tokenCnt = cnt; }
  // Get the token string by index
  std::string getStr(size_t idx) {
    std::string str = this->idToTokenVec[idx];
    return str;
  }

private:
  // Check if a character is a whitespace character.
  bool isWhitespace(char s) const {
    // TODO-HIGH: Consider using standard library functions like `isspace`.
    // return isspace(static_cast<unsigned char>(s));
    return s == ' ' || s == '\t' || s == '\n' || s == '\r';
  }
  // Check if a character is a punctuation character.
  bool isPunctuation(char s) const {
    // TODO-HIGH: Consider using standard library functions like `ispunct`.
    // return ispunct(static_cast<unsigned char>(s));
    return (s >= 33 && s <= 47) || (s >= 58 && s <= 64) ||
           (s >= 91 && s <= 96) || (s >= 123 && s <= 126);
  }
  // Change character from uppercase to lowercase
  char toLower(char s) const {
    // TODO-HIGH: Consider using standard library functions like `tolower`.
    // return static_cast<char>(tolower(static_cast<unsigned char>(s)));
    if (s >= 65 && s <= 90)
      return s + 32;
    else
      return s;
  }
  // Check if a char is a chinese character
  // TODO-MID: find more accurate strategy and write more comments.
  bool isChineseChar(char s) {
    const size_t lookup[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4};
    int8_t highbits = static_cast<uint8_t>(s) >> 4;
    return lookup[highbits] == 3;
  }
  // Replace all " " with "▁"
  std::string replaceAllSpace(const std::string &str) {
    std::string res;
    int index = 0;
    std::string replace = "▁";
    res.append(replace);
    for (char c : str) {
      if (c != ' ') {
        res.push_back(c);
      }
      if (c == ' ' && (index == 0 || str[index - 1] != ' ')) {
        res.append(replace);
      }
      index++;
    }
    return res;
  }
  // Process a token and store its corresponding value in the container.
  // This function takes a token as input and find its corresponding value in
  // the token-to-id map.
  // The option affix decides if function tokenize string by affix.
  // The longest string matching strategy is adopted for word segmentation using
  // root affixes. If the root affix method is not used, the function adopt the
  // following strategy: If the token exists in the map, the corresponding value
  // is stored in the container at the current token count index. If the token
  // is not found in the map, the value 100 (corresponding to the unknown token
  // [UKN]) is stored in the container. Finally, the token count is incremented.
  void processToken(const std::string &token, size_t &tokenCnt,
                    bool affix = false);
  void tokenizeWithAffix(const std::string &token, size_t &tokenCnt);
  std::string findLongestSubToken(const std::string &token, size_t start);
  void assignTokenId(const std::string &token, size_t &tokenCnt);
  // Load vocab into class
  void loadVocab(const std::string &token);
  // [UNK] NLP Padding Marker
  int pad;
  // [UNK] NLP Unknown Marker
  int unk;
  // [CLS] NLP Classification Marker
  int cls;
  // [SEP] NLP Separator Marker
  int sep;
  // The maximum number of input characters that can be accepted in one word.
  size_t maxInputChars = 200;
  // The string member of the text container.
  std::string str;
  // Token-ID map holds the given vocabulary.
  // Since the map is only used for quick lookups and not iterating through it
  // in a specific order, using `std::unordered_map` for faster lookups.
  std::unordered_map<std::string, size_t> tokenToIdMap;
  // ID-Token vector holds the given vocabulary.
  // It is faster to find elements by index.
  std::vector<std::string> idToTokenVec;
  // Record token count.
  size_t tokenCnt;
};

// Text Constructor with string.
template <typename T, size_t N>
Text<T, N>::Text(const std::string &str) : MemRef<T, N>(), str(str) {}

// LLaMA Tokenizer
template <typename T, size_t N>
void Text<T, N>::tokenizeLlama(const std::string &vocab, size_t length) {
  // Initialize MemRef container members.
  this->offset = 0;
  this->sizes[0] = 1;
  this->sizes[1] = length;
  this->setStrides();
  this->size = this->product(this->sizes);
  this->allocated = new T[this->size];
  this->aligned = this->allocated;
  this->unk = 0;
  this->cls = 1;
  this->sep = 2;
  this->pad = 2;
  // Load Vocab
  loadVocab(vocab);
  str = replaceAllSpace(str);

  int len = str.length();
  std::vector<size_t> res;
  std::vector<float> score(len + 1, 0);
  std::vector<size_t> prev(len + 1, 0);
  // Reserve space for the results.
  res.reserve(len);

  // Forward pass
  // Use dynamic programming as the main algorithm to adapt the longest
  // charactors.
  for (int i = 0; i < len; i++) {
    for (int sub_len = 1; sub_len <= len - i; sub_len++) {
      auto iter_start = str.begin() + i;
      auto iter_end = iter_start + sub_len;
      auto token = tokenToIdMap.find(std::string(iter_start, iter_end));
      if (token != tokenToIdMap.end()) {
        int token_score = sub_len * sub_len;
        int local_score = score[i] + token_score;
        int next = i + sub_len;
        if (score[next] < local_score) {
          score[next] = local_score;
          prev[next] = token->second;
        }
      }
    }
  }
  // Backward pass
  int i = len;
  while (i > 0) {
    size_t token_id = prev[i];
    res.push_back(token_id);
    i -= idToTokenVec[token_id].length();
  }

  this->aligned[0] = cls;
  tokenCnt = 1;

  // Directly fill this->aligned in reverse order.
  for (auto it = res.rbegin(); it != res.rend(); ++it) {
    this->aligned[tokenCnt++] = *it;
  }

  for (size_t i = tokenCnt; i < length; i++) {
    this->aligned[i] = pad;
  }
}

// Bert Tokenizer
template <typename T, size_t N>
void Text<T, N>::tokenizeBert(const std::string &vocab, size_t length,
                              bool lower, bool affix) {
  // Initialize MemRef container members.
  this->offset = 0;
  this->sizes[0] = 1;
  this->sizes[1] = length;
  this->setStrides();
  this->size = this->product(this->sizes);
  this->allocated = new T[this->size];
  this->aligned = this->allocated;
  this->pad = 0;
  this->unk = 100;
  this->cls = 101;
  this->sep = 102;
  loadVocab(vocab);
  // Tokenize string and convert to MemRef container object.
  // Mark the beginning of our token.
  this->aligned[0] = cls;
  tokenCnt = 1;
  std::string token;
  for (size_t i = 0; i < str.size(); i++) {
    char s = str[i];
    if (lower) {
      s = toLower(s);
    }
    if (isWhitespace(s) || isPunctuation(s) || isChineseChar(s)) {
      if (!token.empty()) {
        processToken(token, tokenCnt, affix);
        token.clear();
      }
      if (isPunctuation(s)) {
        token = s;
        processToken(token, tokenCnt, false);
        token.clear();
      }
      if (isChineseChar(s)) {
        token.append(str, i, 3);
        // If it doesn't divide by affix, divide the Chinese words one by one.
        if (!affix) {
          processToken(token, tokenCnt, false);
          token.clear();
        }
        i += 2;
      }
    } else {
      token += s;
    }
  }

  // Parse the last token if exists.
  if (!token.empty()) {
    processToken(token, tokenCnt, affix);
  }

  // Mark the end of token stream.
  this->aligned[tokenCnt++] = sep;
  // Padding the rest text container.
  for (size_t i = tokenCnt; i < length; i++) {
    // TODO-HIGH: considering use `pad` here.
    this->aligned[i] = sep;
  }
}

// TODO-HIGH: consider using `revertLlama` here.
template <typename T, size_t N>
std::string Text<T, N>::revert(Text<T, 2> input) {
  std::string dst;

  const int PAD_ID = 0;
  const int CLS_ID = 1;
  const int SEP_ID = 2;

  for (size_t i = 0; i < this->size; i++) {
    int id = input.getData()[i];
    if (id == PAD_ID || id == CLS_ID)
      continue;
    if (id == SEP_ID)
      break;
    std::string token = this->idToTokenVec[id];
    if (token.find("▁") != std::string::npos) {
      dst.append(" ");
      // TODO-HIGH: consider whether the `3` is reasonable here.
      dst.append(token, 3);
    } else {
      dst.append(token);
    }
  }
  if (dst[0] == ' ') {
    dst.erase(0, 1);
  }
  return dst;
}

template <typename T, size_t N>
void Text<T, N>::loadVocab(const std::string &vocab) {
  // TODO-LOW: If in the future, there are more vocab file types to support,
  // consider implementing a more advanced mechanism to determine
  // and process each file type.
  std::ifstream fin(vocab);
  if (!fin.is_open()) {
    throw std::runtime_error("Failed to open vocab file: " + vocab);
  }

  std::string token;
  size_t index = 0;

  while (getline(fin, token)) {
    tokenToIdMap[token] = index++;
    idToTokenVec.push_back(token);
  }
  fin.close();
}

template <typename T, size_t N>
void Text<T, N>::processToken(const std::string &token, size_t &tokenCnt,
                              bool affix) {
  if (affix) {
    tokenizeWithAffix(token, tokenCnt);
  } else {
    assignTokenId(token, tokenCnt);
  }
}

template <typename T, size_t N>
void Text<T, N>::tokenizeWithAffix(const std::string &token, size_t &tokenCnt) {
  if (token.size() > maxInputChars) {
    this->aligned[tokenCnt++] = unk;
    return;
  }
  size_t start = 0;
  while (start < token.size()) {
    std::string subToken = findLongestSubToken(token, start);
    if (subToken.empty()) {
      this->aligned[tokenCnt++] = unk;
      return;
    }
    this->aligned[tokenCnt++] = tokenToIdMap[subToken];
    start += subToken.size();
  }
}

template <typename T, size_t N>
std::string Text<T, N>::findLongestSubToken(const std::string &token,
                                            size_t start) {
  size_t end = token.size();
  while (start < end) {
    std::string substr = token.substr(start, end - start);
    if (start > 0) {
      substr = "##" + substr;
    }
    if (tokenToIdMap.count(substr)) {
      return substr;
    }
    end--;
  }
  return "";
}

template <typename T, size_t N>
void Text<T, N>::assignTokenId(const std::string &token, size_t &tokenCnt) {
  if (tokenToIdMap.count(token)) {
    this->aligned[tokenCnt++] = tokenToIdMap[token];
  } else {
    this->aligned[tokenCnt++] = unk;
  }
}
} // namespace buddy

#endif // FRONTEND_INTERFACES_BUDDY_LLM_TEXTCONTAINER
