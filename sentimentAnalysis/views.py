from django.shortcuts import render
from django.http import JsonResponse
from .load_models import vectorizer, LinearSVC, LogisticRegression, MultinomialNaiveBayes, RandomForest, DecisionTree
import pickle

def index(request):
    return render(request, 'index.html')

def analyze_sentiment(reviewText):

    # Vectorizing the text for prediction
    review_vector = vectorizer.transform([reviewText])

    # Predict using different models
    linearSVC = LinearSVC.predict(review_vector)[0]
    logisticReg = LogisticRegression.predict(review_vector)[0]
    multinomialNaiveBayes = MultinomialNaiveBayes.predict(review_vector)[0]
    randomForest = RandomForest.predict(review_vector)[0]
    decisionTree = DecisionTree.predict(review_vector)[0]

    return {
        'Linear_SVC': linearSVC,
        'Logistic_Regression': logisticReg,
        'Multinomial_NaiveBayes': multinomialNaiveBayes,
        'Random_Forest': randomForest,
        'Decision_Tree': decisionTree,
    }


def analyze(request):
    if request.method == 'POST':
        reviewText = request.POST['reviewText']
        sentiments = analyze_sentiment(reviewText)
    return render(request, 'result.html', {'sentiments': sentiments})

    # vectorizer
    # text = [review_text]

    # vectorizedText = vectorizer.transform(text)

    # prediction = LinearSVC.predict(vectorizedText)

    # prediction_result = prediction[0]

    # context = {
    #     'prediction_result': prediction_result
    # }

    # return render(request, 'result.html', context)