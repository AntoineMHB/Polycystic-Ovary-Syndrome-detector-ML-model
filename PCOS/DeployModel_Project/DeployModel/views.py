from django.http import HttpResponse
from django.shortcuts import render
import joblib

def home(request):
    return render(request, "home.html")

def result(request):

    cls = joblib.load('ThePCOS_model.joblib')

    lis = []

    lis.append(request.GET['Age_(yrs)'])
    lis.append(request.GET['Weight_(Kg)'])
    lis.append(request.GET['Height(Cm)'])
    lis.append(request.GET['BMI'])
    lis.append(request.GET['Blood_Group'])
    lis.append(request.GET['Pulse_rate(bpm)'])
    lis.append(request.GET['RR_(breaths/min)'])
    lis.append(request.GET['Hb(g/dl)'])
    lis.append(request.GET['Cycle(R/I)'])
    lis.append(request.GET['Cycle_length(days)'])
    lis.append(request.GET['Marraige_Status_(Yrs)'])
    lis.append(request.GET['Pregnant(Y/N)'])
    lis.append(request.GET['No_of_abortions'])
    lis.append(request.GET['I_beta_HCG(mIU/mL)'])
    lis.append(request.GET['II_beta_HCG(mIU/mL)'])
    lis.append(request.GET['FSH(mIU/mL)'])
    lis.append(request.GET['LH(mIU/mL)'])
    lis.append(request.GET['FSH/LH'])
    lis.append(request.GET['Hip(inch)'])
    lis.append(request.GET['Waist(inch)'])
    lis.append(request.GET['Waist_Hip_Ratio'])
    lis.append(request.GET['TSH(mIU/L)'])
    lis.append(request.GET['PRL(ng/mL)'])
    lis.append(request.GET['Vit_D3(ng/mL)'])
    lis.append(request.GET['PRG(ng/mL)'])
    lis.append(request.GET['RBS(mg/dl)'])
    lis.append(request.GET['Weight_gain(Y/N)'])
    lis.append(request.GET['hair_growth(Y/N)'])
    lis.append(request.GET['Reg_Exercise(Y/N)'])

    lis.append(request.GET['Endometrium(mm)'])
    lis.append(request.GET['Skin_darkening(Y/N)'])
    lis.append(request.GET['BP_Diastolic(mmHg)'])
    lis.append(request.GET['Hair_loss(Y/N)'])

    #print(lis)

    ans = cls.predict([lis])

    # if(ans == 1):{
    #     ans == 'Positive';
    # }
    # else:{
    #     ans == 'Negative'
    # }

    return render(request, "result.html", {'ans':ans})
    