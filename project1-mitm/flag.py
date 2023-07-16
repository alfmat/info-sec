# MITM Project - New Flag2
#
# Author: Joseph Abel
# Email:  jabel9@gatech.edu
#
#
# Generate a unique hash (sha256) given a gtID and display on the screen 
#

# gt is 903401672

import hashlib

print("Enter your GTID")
x = input("") # don't put anything in the quotes enter your georgia tech number in prompt
hash_object = hashlib.sha256(x.encode())
hex_dig = hash_object.hexdigest()
hash_object2 = hashlib.sha256(b"CS6035-Flag2R1PR0A3R")
hex_dig22 = hash_object2.hexdigest()
print("Hash    :  ", hex_dig+hex_dig22)