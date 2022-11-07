import mysql.connector

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="JackLai+19980510",
  database="mydatabase"
)
mycursor = mydb.cursor()


# for x in mycursor:
#   print(x)