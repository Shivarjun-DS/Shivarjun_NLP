for simplicity We have used TF-IDF vectorizer over Bag of words and Bag of n-grams to solve this problem. because bag of words and Bag of n-grams models treats all texts equally. were as in TF-IDF gives importance of a given word relative to other words in the document and in the corpus.
We have use TF-IDF also because it is bet suited for Information Retrieval and TF-IDF is popular scheme for initial versions of solutions.

Prediction using PC
1) open VScode 
2) enter command 'git clone git@github.com:Shivarjun-DS/Shivarjun_NLP.git'
3) enter command 'python NLP.py'
4) It will ask for your input enter the product, store or Brand to get top 10 suggestions

Prediction Using Docker
1) Install Docker Desktop on PC
2) open VScode and enter the command 'docker pull shivarjun/nlp:latest'
3) enter command 'docker run -it --rm shivarjun/nlp'
4) It will ask for your input enter the product, store or Brand to get top 10 suggestions