
from hashlib import md5
from django.shortcuts import render, redirect
from django.http import HttpResponse
from login.models import Question, Student
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt
from MapDisplay import views

# Create your views here.


@login_required(login_url='login/')
def index(request):
    htmlCode = '<table border="1px">'
    for row in Question.objects.all():
        htmlCode += ('<tr><td>'+str(row.id)+'</tr><td>' +
                     row.questionContent+'</td><td>'+row.answer+'</td>')
    htmlCode += '</table>'
    return HttpResponse(htmlCode)


@csrf_exempt
def signin(request):
    request.session = {}
    views.plotting(request)
    try:
        account = request.POST['usr']
        password = md5(request.POST['pwd'].encode()).hexdigest()
        user = Student.objects.filter(account=account, password=password)
        if user:
            request.session['account'] = account
            request.session['name'] = user[0].name
            return redirect('/display/map')
        else:
            return render(request, 'Register_Page.html',  {'msg': 'Incorrect username or password'})
    except Exception:
        return render(request, 'Register_Page.html', {'msg': None})


def register(request):
    try:
        post = request.POST
        account = post.get('account')
        email = post.get('email')
        if Student.objects.filter(account=account):
            return render(request, 'registerfunction.html', {'msg': "user name exists"})
        if Student.objects.filter(email=email):
            return render(request, 'registerfunction.html', {'msg': "email has been used"})
        password = post.get('pwd1')
        p2 = post.get('pwd2')
        if password != p2:
            return render(request, 'registerfunction.html', {'msg': 'Two inconsistent password entries'})
        name = post.get('name')
        email = post.get('email')
        password = md5(password.encode()).hexdigest()
        s = Student(account=account, password=password, name=name, email=email)
        s.save()
        return render(request, 'Register_Page.html', {'msg': 'Resgister successful!'})
    except Exception:
        return render(request, 'registerfunction.html', {'msg': None})


def reset(request):
    try:
        post = request.POST
        account = post.get('account1')
        email = post.get('email1')
        if not Student.objects.filter(account=account):
            return render(request, 'reset.html', {'msg': "Please enter your account information!"})
        user2 = Student.objects.filter(account=account, email=email)
        if user2:
            password = post.get('npwd')
            password = md5(password.encode()).hexdigest()
            user1 = Student.objects.get(account=account)
            user1.password = password
            user1.save()
            return redirect(request, 'Register_Page.html', {'msg': 'Reset successful!'})
        else:
            return render(request, 'reset.html', {'msg': "The account does not match the registered email address"})
    except Exception:
        return render(request, 'registerfunction.html', {'msg': None})
