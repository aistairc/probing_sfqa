. probing/bin/activate

for TARGET in "FBQ" "SQ" "WQ"
do
  DATA=$(pwd)/simple-qa-analysis/datasets/$TARGET
  for MODEL in $(cat BertQA/list.txt);
  do
    RES=$TARGET/$MODEL

    #entity detection
    cd BertQA
    python main.py --data_dir $DATA --type ed --save_path ./$TARGET --model $MODEL --finetune

    #entity linking
    cd ../BuboQA/entity_linking
    python entity_linking.py --model_type lstm --query_dir ../../BertQA/$TARGET/$MODEL --data_dir $DATA --output_dir ./$RES

    #relation prediction
    cd ../../BertQA
    python main.py --data_dir $DATA --type rp --save_path ./$TARGET --model $MODEL --finetune

    #evidence integration
    cd ../BuboQA/evidence_integration
    python evidence_integration.py --ent_type lstm --ent_path ../entity_linking/$RES/lstm/test-h100.txt --rel_type cnn --rel_path ../../BertQA/$TARGET/$MODEL/test.txt --data_path $DATA/test.txt --output_dir ./$RES

    cd ../..

  done
done

deactivate

