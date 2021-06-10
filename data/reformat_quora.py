with open('quora/quora_duplicate_questions.tsv') as f_in, open('quora/quora_questions.tsv','w') as f_out:
    for line in f_in:
        try:
            source, target = line.split('\t')[3],line.split('\t')[4]
        except:
            print(line)
        f_out.write( '{}\t{}\n'.format(source,target))