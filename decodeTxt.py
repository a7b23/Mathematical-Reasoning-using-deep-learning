import numpy as np

file_name = "sample.txt"

f = open(file_name,'r')

data = f.readlines()

input_vocab = ['What','Sum','Units','Tens','Max','Value','digit','given','is','of','in','x','y','+','-','?','>','<','odd','even','<unk>']
output_vocab = ['0','1','2','3','4','5','6','7','8','9','True','False','<s>','</s>','<append>','+','-']

word_dict = {}
for idx,word in enumerate(input_vocab) :
    word_dict[word.lower()] = idx

for line in data :
	words = line.split(',')
	question = ""
	answer = ""
	question_part = words[1:9]
	answer_part = words[9:]

	for q in question_part :
		if input_vocab[int(q)] == '<unk>' :
			break
		question +=input_vocab[int(q)] +" "
	
	for ans in answer_part :
		if output_vocab[int(ans)] == '</s>' :
			break
		elif not output_vocab[int(ans)] == '<s>' :
			answer += output_vocab[int(ans)]

	print('the image is %s'%words[0])
	print('the corresponding question is :- %s'%question)
	print('the corresponding expected answer is :- %s'%answer)
	print('\n')

