Annotated files are present under the `annotated/` folder. Each chapter file contains annotations for a particular chapter stored in a comma separated values (.csv) format. Each row in the csv contains a token. Each sentence in the file is row-separated. Each row contains the `Word`, `Target`, `Scene role`,  `Function`, and `MWE` fields . The details for these field are given in the table below.

| __Field__ | __Explanation__ | 
| :---:   | :---: | 
| Word |  Contains a space-separated token in the sentence|
| Target | Indicates the target in the token (or, the complete token) to be annotated. The target is not normalized for gender and number infelctions|
| Scene role | The Scene Role label for the target according to the SNACS hierarchy|
| Function | The Function label for the target according to the SNACS hierarchy|
| MWE | Multi-word Expression. Indicates whether a target is multi-word or not. Each token in a MWE is indicated in the format `a:b` where a is the instance number of the MWE in the sentence, and b indicates the index of the token within the MWE.| 

Note that `Target`, `Scene role`, and `Function` fields are only filled for tokens which are/contain case markers or adpositions. In the case of MWE, only fields corresponding to the first token contain the `Scene Role` and `Function` labels.

In addition, there is a `Sentence_id` field which has a sentence ID unique to the chapter filled in the row containing the first token of the sentence.

The `Transliteration`,`Gloss`, and `Notes` fields can be ignored as these are just meant to help a reader understand a few examples and are not available for all chapters.
