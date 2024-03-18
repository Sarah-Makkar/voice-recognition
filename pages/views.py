# pages/views.py
from django.shortcuts import render, HttpResponseRedirect
from django.http import Http404
from django.urls import reverse
from django.views.generic import TemplateView


def homePageView(request):
    return render(request, 'home.html')


from django.http import HttpResponseRedirect
from django.urls import reverse


def homePost(request):
    # Use request object to extract choice.
    try:
        # Extract value from request object by control name.
        sdStr = request.POST['sd']
        Q25Str = request.POST['Q25']
        IQRStr = request.POST['IQR']
        spEntStr = request.POST['spEnt']
        sfmStr = request.POST['sfm']
        meanfunStr = request.POST['meanfun']

        # Crude debugging effort.
        print("*** sd: " + str(sdStr))
        sd = float(sdStr)
        Q25 = float(Q25Str)
        IQR = float(IQRStr)
        spEnt = float(spEntStr)
        sfm = float(sfmStr)
        meanfun = float(meanfunStr)
    # Enters 'except' block if integer cannot be created.
    except:
        return render(request, 'home.html', {
            'errorMessage': '*** The data submitted is invalid. Please try again.'
        })
    else:
        # Always return an HttpResponseRedirect after successfully dealing
        # with POST data. This prevents data from being posted twice if a
        # user hits the Back button.
        return HttpResponseRedirect(reverse('results', kwargs={'sd': sd, 'Q25': Q25, 'IQR': IQR,
                                                               'spEnt': spEnt, 'sfm': sfm,
                                                               'meanfun': meanfun}))


import pickle
import sklearn
import pandas as pd


def results(request, sd, Q25, IQR, spEnt, sfm, meanfun):
    # load saved model
    with open('./model_pkl', 'rb') as f:
        loadedModel = pickle.load(f)
    # Create a single prediction.
    singleSampleDf = pd.DataFrame(columns=['sd', 'Q25', 'IQR', 'sp.ent', 'sfm', 'meanfun'])

    singleSampleDf = singleSampleDf.append({'sd': sd, 'Q25': Q25, 'IQR': IQR,
                                            'sp.ent': spEnt, 'sfm': sfm,
                                            'meanfun': meanfun},
                                           ignore_index=True)

    cols = singleSampleDf.columns
    singleSampleDf[cols] = singleSampleDf[cols].apply(pd.to_numeric)

    singlePrediction = loadedModel.predict(singleSampleDf)
    print("Single prediction: " + str(singlePrediction))

    return render(request, 'results.html', {'sd': sd, 'Q25': Q25, 'IQR': IQR,
                                            'spEnt': spEnt, 'sfm': sfm,
                                            'meanfun': meanfun, 'prediction': singlePrediction})
