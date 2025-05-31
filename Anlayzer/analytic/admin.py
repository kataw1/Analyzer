from django.contrib import admin
from .models import Premium_Service, Service , Contact , Favorite


# Register your models here.

admin.site.register(Service)
admin.site.register(Contact)
admin.site.register(Favorite)
admin.site.register(Premium_Service)



admin.site.site_url = "/home" # this to make the main website the home page when u press on viewsite