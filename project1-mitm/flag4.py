#run this program and your input should be your gtid i.e. jabel9 make sure python version is >=3.x
#don't change any variables or alter the integrity of the program failure to comply with these instructions will result in a 0 for this flag
#name the file to flag4.py and run it as python flag4.py or python3 flag4.py
import hashlib
x = input("") #georgia tech ID
hash_object = hashlib.sha256(x.encode())
hex_dig = hash_object.hexdigest()
hash_object2 = hashlib.sha256(b"CS6035-Flag4l33thax0r")
hex_dig22 = hash_object2.hexdigest()
print("Combiend hash    :  ", hex_dig+hex_dig22)