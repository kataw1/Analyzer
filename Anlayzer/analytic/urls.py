from django.contrib import admin
from django.urls import path
from .views import aboutus, activate, add_to_fav, color_duplicates, contact, fill_empty_data, login, logout, payment, predict_from_excel, predict_profit, predict_success, premium_services, remove_from_fav, search_service, service_filter, sort_excel_column , mean_mode_median,services,concatenate,home,remove_duplicates,remove_duplicates_rows,register, summarize_pdf, view_fav
from django.contrib.auth import views as auth_views # for password and access built in forms



urlpatterns = [
    #__WebSite_Urls__#
    path('home/', home, name='home'),
    path('services/', services, name='services'),
    path('contact', contact, name='contact'),
    path('aboutus/',aboutus,name='aboutus'),
    path('payment/',payment,name='payment'),
    path('premium-services/',premium_services,name='premium-services'),




    #__Ai_ToolsUrls__#
    path('predict/', predict_success, name='predict_success'),
    path('predict2/', predict_from_excel, name='predict_from_excel'),
    path('summarize-pdf/',summarize_pdf, name='summarize_pdf'),
    path('accounts/login/', auth_views.LoginView.as_view(template_name="login.html"), name='login'),


    
    #__Analysis_ToolsUrls__#
    path('calculate/', mean_mode_median, name='mean_mode_median'),
    path('sort/', sort_excel_column, name='sort_excel_column'),
    path('concatenate/', concatenate, name='concatenate'),
    path('clean-data/', remove_duplicates, name='remove_duplicates'),
    path('clean-data-rows/', remove_duplicates_rows, name='remove_duplicates_rows'),
    path('predict_profit/', predict_profit, name='predict_profit'),
    path('color-duplicates/', color_duplicates, name='color_duplicates'),



    #__Regestration Url__#
    path('login', login, name='login'),
    path('register', register, name='register'),
    path('logout',logout,name='logout'),
    
    #__Active_Email__#
    path('activate/<str:token>/', activate, name='activate'),

    
    #__ResetPassword__#
    path("password_reset/", auth_views.PasswordResetView.as_view(template_name = "reset_pass.html"), name="password_reset"), # we use built in , then we use template name method to access the temp name to customize and protect admin panel
    path("password_reset/done/",auth_views.PasswordResetDoneView.as_view(template_name = "reset_pass_sent.html"),name="password_reset_done",),
    path("reset/<uidb64>/<token>/",auth_views.PasswordResetConfirmView.as_view(template_name = "reset_pass_form.html"),name="password_reset_confirm",),
    path("reset/complete/",auth_views.PasswordResetCompleteView.as_view(template_name = "reset_pass_done.html"),name="password_reset_complete",),

    
    #__UserFunctions__#
    path('add-to-fav/<int:service_id>/',add_to_fav, name='add_to_fav'),
    path('remove-from-fav/<int:service_id>/', remove_from_fav, name='remove_from_fav'),
    path('view-fav', view_fav, name='view-fav'),
    path('search/', search_service, name='search_service'),
    path('filter/', service_filter, name='service_filter'),
    path('fill-empty/', fill_empty_data, name='fill_empty_data'),


]
