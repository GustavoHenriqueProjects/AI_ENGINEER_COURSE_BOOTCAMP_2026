# IMPORTANTE: Execute a célula acima primeiro para definir a variável 'sentence'
sentence = "Her cat's name is Luma"
lower_sentence = sentence.lower()
print(lower_sentence)

setence_list = ['Could you pass me the tv remote',
                'It is IMPOSSIBLE to find this hotel', 
                'What is the weather in Tokyo']

lower_sentence_list = [x.lower() for x in setence_list]

print(lower_sentence_list)

