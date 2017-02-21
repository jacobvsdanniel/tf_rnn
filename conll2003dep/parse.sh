#!/bin/bash

PARSER_EVAL=/home/danniel/Downloads/tf_models/syntaxnet/bazel-bin/syntaxnet/parser_eval
MODEL_DIR=/home/danniel/Downloads/tf_models/syntaxnet/syntaxnet/models/parsey_mcparseface
TASK_SPEC=/home/danniel/Desktop/rnn_ner/conll2003dep/task.pbtxt

$PARSER_EVAL \
  --input=conll2003_sentence_$1 \
  --output=stdout-conll \
  --hidden_layer_sizes=64 \
  --arg_prefix=brain_tagger \
  --graph_builder=structured \
  --task_context=$TASK_SPEC \
  --model_path=$MODEL_DIR/tagger-params \
  --slim_model \
  --batch_size=1024 \
   | \
  $PARSER_EVAL \
  --input=stdin-conll \
  --output=conll2003_parse_$1 \
  --hidden_layer_sizes=512,512 \
  --arg_prefix=brain_parser \
  --graph_builder=structured \
  --task_context=$TASK_SPEC \
  --model_path=$MODEL_DIR/parser-params \
  --slim_model \
  --batch_size=1024 \
  
  
  
  
  
  
