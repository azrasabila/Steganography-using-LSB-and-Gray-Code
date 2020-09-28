import cv2
import numpy as np
import math

def to_bin(data):
	"""Convert `data` to binary format as string"""
	if isinstance(data, str):
		return ''.join([ format(ord(i), "08b") for i in data ])
	elif isinstance(data, bytes) or isinstance(data, np.ndarray):
		return [ format(i, "08b") for i in data ]
	elif isinstance(data, int) or isinstance(data, np.uint8):
		return format(data, "08b")
	else:
		raise TypeError("Type not supported.")

# Helper function to xor two characters 
def xor_c(a, b): 
	return '0' if(a == b) else '1'
  
# Helper function to flip the bit 
def flip(c): 
	return '1' if(c == '0') else '0'
  
# function to convert binary string 
# to gray string 
def binary_to_gray(binary): 
	gray = ""
  
	# MSB of gray code is same as 
	# binary code 
	gray += binary[0]
  
	# Compute remaining bits, next bit  
	# is comuted by doing XOR of previous  
	# and current in Binary 
	for i in range(1,len(binary)): 
		  
		# Concatenate XOR of previous  
		# bit with current bit 
		gray += xor_c(binary[i - 1],  
					  binary[i])
  
	return gray
  
# function to convert gray code 
# string to binary string 
def gray_to_binary(gray): 
  
	binary = ""
  
	# MSB of binary code is same  
	# as gray code 
	binary += gray[0]
  
	# Compute remaining bits 
	for i in range(1, len(gray)): 
		  
		# If current bit is 0,  
		# concatenate previous bit 
		if (gray[i] == '0'): 
			binary += binary[i - 1]
  
		# Else, concatenate invert  
		# of previous bit 
		else: 
			binary += flip(binary[i - 1])
  
	return binary

def encodeMessage(image, secret_data):
	# read the image
	#image = cv2.imread(image_name)
	# maximum bytes to encode
	n_bytes = image.shape[0] * image.shape[1] * 3 // 8
	print("[*] Maximum bytes to encode:", n_bytes)
	if len(secret_data) > n_bytes:
		raise ValueError("[!] Insufficient bytes, need bigger image or less data.")
	print("[*] Encoding data...")
	print(secret_data)
	data_index = 0
	# add stopping criteria
	secret_data += "====="
	# convert data to binary
	binary_secret_data = to_bin(secret_data)
	#print("data in binary :" + binary_secret_data)
	#convert ke gray
	gray_secret_data = binary_to_gray(binary_secret_data)
	#print("data in gray	  :" + gray_secret_data)
	# size of data to hide
	data_len = len(gray_secret_data)
	for row in image:
		for pixel in row:
			#if edged[i-1][j-1] == 1 :
			# convert RGB values to binary format
			r, g, b = to_bin(pixel)
			# modify the least significant bit only if there is still data to store
			if data_index < data_len:
				# least significant red pixel bit
				pixel[0] = int(r[:-1] + gray_secret_data[data_index], 2)
				data_index += 1
			if data_index < data_len:
				# least significant green pixel bit
				pixel[1] = int(g[:-1] + gray_secret_data[data_index], 2)
				data_index += 1
			if data_index < data_len:
				# least significant blue pixel bit
				pixel[2] = int(b[:-1] + gray_secret_data[data_index], 2)
				data_index += 1
			# if data is encoded, just break out of the loop
			if data_index >= data_len:
				break
	return image

def decodeMessage(image):
	print("[+] Decoding...")
	# read the image
	#image = cv2.imread(image_name)
	binary_data = ""
	for row in image:
		for pixel in row:
			r, g, b = to_bin(pixel)
			binary_data += r[-1]
			binary_data += g[-1]
			binary_data += b[-1]
	#print(binary_data)
	#print(gray_to_binary(binary_data))
	binary_data = gray_to_binary(binary_data)
	# split by 8-bits
	all_bytes = [ binary_data[i: i+8] for i in range(0, len(binary_data), 8) ]
	# convert from bits to characters
	decoded_data = ""
	for byte in all_bytes:
		decoded_data += chr(int(byte, 2))
		#print(decoded_data)
		if decoded_data[-5:] == "=====":
			break
	return decoded_data[:-5]

def psnr(img1, img2):	
	# img1 and img2 have range [0, 255]
	#img1 = img1.astype(np.float64)
	#img2 = img2.astype(np.float64)
	#print(img1)
	#print(img2)
	img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
	mse = np.mean(img1 - img2)/(img1.shape[0] * img1.shape[1])
	print("mse : ")
	print("{:12.10f}".format(mse))
	if mse == 0:
		return float('inf')
	return 20 * math.log10(255.0 / math.sqrt(mse))

def main():
	img = cv2.imread('pool.png')
	#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	#cv2.imshow('original',img)

	f = open("length50.txt", "r")
	text = f.read()
	#print(text)

	stegoImage = encodeMessage(img, text)
	cv2.imshow('output', stegoImage)
	#stegoImage2 = encodeMessage(img, text2)

	cv2.imwrite('pool1000.png', stegoImage)

	psnr1 = psnr(img, stegoImage)
	print("psnr : ")
	print ("{:12.10f}".format(psnr1))

	secretMessage = decodeMessage(stegoImage)
	print(secretMessage)

	cv2.waitKey(0)
	cv2.destroyAllWindows()

if __name__ == "__main__":
	main()