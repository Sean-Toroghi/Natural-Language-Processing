

# Preprocessing text - Cleaning
def handle_non_ascii(text):
    text = text.replace('\xa0', ' ')
    text = re.sub("\x91|\x92|\xb4|‘|’", '\'', text)     
    text = re.sub("\x93|\x94|\xa8|“|”", '"', text)     
    text = re.sub("\x97|—", '-', text)     
    text = re.sub(r"CO(\n|\s)*\xb2", "CO2", text)
    text = re.sub("\xb2|\xb9|̈́", '', text)
    text = re.sub("\xf3|\xd3|\xd6", 'o', text)       
    text = re.sub("\xa9|\x99|\xae", '', text)         
    text = re.sub('\xb6', "para ", text)              
    text = re.sub('\xe4|\xe5|\xe1', 'a', text)        
    text = re.sub('ś', "'s", text)                    
    text = re.sub('\xe9', 'e', text)                 
    text = re.sub('\xcb', 'E', text)                 
    text = re.sub('\xfe', 't', text)                

    return text

cList = {
  "ain't": "am not", "aren't": "are not", "can't": "cannot", "can't've": "cannot have", "'cause": "because",
  "could've": "could have", "couldn't": "could not", "couldn't've": "could not have", "didn't": "did not",
  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hadn't've": "had not have", "hasn't": "has not",
  "haven't": "have not", "he'd": "he would", "he'd've": "he would have", "he'll": "he will", "he'll've": "he will have",
  "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is", "I'd": "I would",
  "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have", "I'm": "I am", "I've": "I have", "isn't": "is not",
  "it'd": "it had", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have", "it's": "it is", "let's": "let us",
  "ma'am": "madam", "mayn't": "may not", "might've": "might have", "mightn't": "might not", "mightn't've": "might not have",
  "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have",
  "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not",
  "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have",
  "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have", "so's": "so is",
  "that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there had", "there'd've": "there would have",
  "there's": "there is", "they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have",
  "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we had", "we'd've": "we would have",
  "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will",
  "what'll've": "what will have", "what're": "what are", "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have",
  "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is",
  "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",
  "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'alls": "you alls", "y'all'd": "you all would",
  "y'all'd've": "you all would have", "y'all're": "you all are", "y'all've": "you all have", "you'd": "you had", "you'd've": "you would have",
  "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have"
}

c_re = re.compile('(%s)' % '|'.join(cList.keys()))

def expandContractions(text, c_re=c_re):
    '''replacing abbreviations'''
    def replace(match):
        return cList[match.group(0)]
    return c_re.sub(replace, text)

def removeHTML(text):
    html = re.compile(r'<.*?>')
    return html.sub(r'',text)

def textPreprocessing(text):
    text = text.strip()
    text = text.replace("World War ||", "World War II")
    text = re.sub(r"\\(?=\s)", '', text)
    text = re.sub(r"(?<=\s)\\", '', text)
    text = re.sub(r"(?<=\s)/(?=\s)", "or", text)
    text = removeHTML(text)   
    text = text.replace("\'\'", '"')    
    text = re.sub(r", ", ",", text)     
    text = re.sub(r"\. ", ".", text)     
    text = re.sub(r"\.{2}", ".", text)     
    text = re.sub(r"\.{4,}", ".", text)     
    text = re.sub(r"\,{2,}", ",", text)     
    text = re.sub(r"(?<=\w)\n{1,}", '', text)   # If next to the newline characters are letters, delete the
    text = re.sub(r"\n{1,}", ' ', text)         # Remove newline characters and replace them with spaces
    text = re.sub(r"\s{2,}", " ", text)         # Replace consecutive spaces with 1 space
    text = handle_non_ascii(text)
    text = expandContractions(text)   
    return text
