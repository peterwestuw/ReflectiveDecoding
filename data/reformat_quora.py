with open('quora/quora_duplicate_questions.tsv') as f_in, open('quora/quora_duplicate_questions.tsv','w') as f_out:
    for line in f_in:
        source, target = line.split('\t')[2],line.split('\t')[3]
        f_out.write( '{}\t{}\n'.format(source,target))