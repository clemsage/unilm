# Data-efficiency of language models
This is a fork of the [Microsoft's UniLM repository](https://github.com/microsoft/unilm) aiming to assess 
the data efficiency of pre-trained language models when fine-tuned for document analysis tasks.

As yet, this codebase is used to fine-tune and evaluate the 
[LayoutLM (v1)](https://github.com/microsoft/unilm/tree/master/layoutlm) model on the Scanned Receipts OCR 
and Information Extraction [(SROIE) benchmark]((https://rrc.cvc.uab.es/?ch=13)) and compare its extraction 
performance with two baseline models that do not leverage pre-training.

For further details, please refer to the inner Readme file, located in the layoutlm folder.

### License
This project is licensed under the license found in the LICENSE file in the root directory of this source tree.
Portions of the source code are based on the [transformers](https://github.com/huggingface/transformers) project.

[Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct)

### Contact Information

For help or issues using this repository, please submit a GitHub issue.

For other communications, please contact Clément Sage (`clement.sage@liris.cnrs.fr`) or 
Thibault Douzon (`thibault.douzon@insa-lyon.fr`).
