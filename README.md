# SemEval 2019 - Task 6 - Identifying and Categorizing Offensive Language in Social Media

# Offensive-Language-Detection
Tweets offensive language detection <br>
Download data from https://sites.google.com/site/offensevalsharedtask/olid<br>

# Results
<br>
<table>
  <tr>
    <th>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</th>
    <th>&nbsp;&nbsp;&nbsp;Model&nbsp;&nbsp;&nbsp;</th>
    <th>&nbsp;&nbsp;&nbsp;Macro F1 Score&nbsp;&nbsp;&nbsp;</th>
  </tr>
  <tr>
    <td>&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;Base Model&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;70&nbsp;&nbsp;&nbsp;</td>
  </tr>
  <tr>
    <td>&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;Glove Embeddings and GRU&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;77&nbsp;&nbsp;&nbsp;</td>
  </tr>
  <tr>
    <td>&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;Glove Embeddings and LSTM&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;77&nbsp;&nbsp;&nbsp;</td>
  </tr>
  <tr>
    <td>&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;Glove Embeddings and biLSTM&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;76&nbsp;&nbsp;&nbsp;</td>
  </tr>
  <tr>
    <td>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;Glove Embeddings and biGRU&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;76&nbsp;&nbsp;&nbsp;</td>
  </tr>
  <tr>
    <td>&nbsp;&nbsp;&nbsp;6&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;Glove biGRU&nbsp;&nbsp;&nbsp;and attention mechanism&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;76&nbsp;&nbsp;&nbsp;</td>
  </tr>
  <tr>
    <td>&nbsp;&nbsp;&nbsp;7&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;GloVe CNN&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;70&nbsp;&nbsp;&nbsp;</td>
  </tr>
  <tr>
    <td>&nbsp;&nbsp;&nbsp;8&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;FastText biGRU&nbsp;&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;72&nbsp;&nbsp;&nbsp;</td>
  </tr>
  <tr>
    <td>&nbsp;&nbsp;&nbsp;9&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;ELMO&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;49&nbsp;&nbsp;&nbsp;</td>
  </tr>
  <tr>
    <td>&nbsp;&nbsp;&nbsp;10&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;Transformer with LSTM&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;73&nbsp;&nbsp;&nbsp;</td>
  </tr>
  <tr>
    <td>&nbsp;&nbsp;&nbsp;11&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;biLSTM&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;70&nbsp;&nbsp;&nbsp;</td>
  </tr>
  <tr>
    <td>&nbsp;&nbsp;&nbsp;12&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;biGRU&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;71&nbsp;&nbsp;&nbsp;</td>
  </tr>
  <tr> <strong>
    <td>&nbsp;&nbsp;&nbsp;<strong>13</strong>&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<strong>Glove FastText</strong>&nbsp;&nbsp;&nbsp;<strong>BiGRU</strong>&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<strong>80</strong>&nbsp;&nbsp;&nbsp;</td> </strong>
  </tr>
</table>

# Source
https://www.kaggle.com/christofhenkel/keras-baseline-lstm-attention-5-fold
https://www.kaggle.com/buchan/transformer-network-with-1d-cnn-feature-extraction
