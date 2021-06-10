mkdir quora

wget https://storage.googleapis.com/ai2-mosaic/public/abductive-commonsense-reasoning-iclr2020/anli.zip 
unzip anli.zip 

wget http://qim.fs.quoracdn.net/quora_duplicate_questions.tsv
cp quora_duplicate_questions.tsv quora
rm quora_duplicate_questions.tsv

python reformat_quora.py
