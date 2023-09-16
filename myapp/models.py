from django.db import models
from .db_conn import db
import datetime

# Create your models here.
current_time = datetime.datetime.now()
name = str(current_time.date())
collection = db[name]