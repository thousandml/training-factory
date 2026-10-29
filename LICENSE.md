CQ SOFTWARE LICENSE

Version 2.0 — Effective October 29, 2026

Copyright (c) 2026 Husam Elfadil, doing business as ThousandML. All rights reserved.

This license governs use of the CQ quantization software, CQ runtime software, Training Factory, compiled native extensions, and associated source code and documentation distributed with this repository (collectively, the "Software").

1. DEFINITIONS

"ThousandML" means Husam Elfadil, doing business as ThousandML, and any successor entity to which the Software and this license are assigned, including any limited liability company or corporation formed to operate the ThousandML business.

(a) "Non-Commercial Use" means use of the Software for personal projects, academic or non-profit research, or internal evaluation of the Software, in each case where neither the Software nor any output of the Software is used to provide a product or service to customers or users, or to support revenue-generating business operations.

(b) "Commercial Use" means any use of the Software that is not Non-Commercial Use, including without limitation:

  (i) quantizing, fine-tuning, or training models for use in a product, service, or business operation;

  (ii) running, serving, or performing inference with any CQ-quantized model, any adapter trained on a CQ-quantized model, or any model that depends on CQ runtime components, in a production, customer-facing, or internal business setting; and

  (iii) providing the Software, or access to its functionality, to third parties.

(c) "Licensed GPU" means a single physical GPU, or a single virtual GPU instance, on which any compiled CQ component executes. Each GPU on a multi-GPU card, server, or cluster counts as a separate Licensed GPU.

(d) "Subscription" means a paid entitlement purchased from ThousandML that authorizes Commercial Use on a stated number of Licensed GPUs for a stated term.

(e) "License Key" means the credential issued by ThousandML that activates a Subscription.

2. NON-COMMERCIAL LICENSE

Subject to the terms of this license, you are granted a limited, non-exclusive, non-transferable, revocable, royalty-free license to:

(a) install and use the Software for Non-Commercial Use;

(b) use the Software to quantize machine learning models into the CQ format for Non-Commercial Use;

(c) fine-tune and run CQ-quantized models with authorized CQ runtime software for Non-Commercial Use; and

(d) modify the Python source code included with the Software for your own internal use.

Internal evaluation of the Software by a business is Non-Commercial Use only while no model trained or served with the Software is used in production or by customers.

3. COMMERCIAL LICENSE

Commercial Use requires an active Subscription.

(a) A Subscription must cover every Licensed GPU used for Commercial Use, whether that GPU is used for quantization, training, fine-tuning, or inference.

(b) Subscriptions are priced per Licensed GPU, or under a separate written enterprise agreement. Current pricing is published by ThousandML at www.thousandml.com or provided on request.

(c) Your Subscription authorizes use of the Software only on hardware that you own, lease, or rent and control. ThousandML does not provide computing hardware under this license.

(d) If your Subscription expires or is terminated, all Commercial Use must stop, including inference with models and adapters produced using the Software. Non-Commercial Use may continue under Section 2.

(e) ThousandML may provide a grace period, offline activation, or an on-premises license server under a Subscription or enterprise agreement.

4. EARLY ACCESS PERIOD

Until ThousandML publicly announces general availability of paid Subscriptions, ThousandML permits Commercial Use of the Software without a Subscription or License Key ("Early Access Period").

The Early Access Period ends thirty (30) days after ThousandML posts a general-availability notice at www.thousandml.com or in this repository. After that date, continued Commercial Use requires an active Subscription.

5. LICENSE KEYS AND VERIFICATION

The Software may require a valid License Key and may periodically verify licensing status, including the number of Licensed GPUs in use. Verification does not transmit your training data, prompts, model outputs, or model weights to ThousandML.

You must not share License Keys outside your organization, or use a License Key on more Licensed GPUs than your Subscription permits.

6. PROPRIETARY COMPONENTS

Certain components of the Software, including CQ quantization algorithms, runtime implementations, native operators, compiled extensions, authentication mechanisms, and related implementation details, are proprietary technology of ThousandML.

These components may be distributed only in compiled or binary form.

Except where expressly permitted by applicable law notwithstanding this restriction, you may not:

(a) reverse engineer, decompile, disassemble, decode, translate, or otherwise attempt to derive the source code, algorithms, implementation, or internal operation of any compiled CQ component;

(b) circumvent, disable, bypass, modify, or interfere with any licensing, authentication, authorization, entitlement, License Key, or access-control mechanism;

(c) extract, reproduce, recreate, or use proprietary CQ algorithms or implementation details to create a competing or substantially similar quantization or inference technology; or

(d) remove, obscure, or alter copyright, trademark, license, attribution, or proprietary notices contained in the Software.

7. REDISTRIBUTION

You may not redistribute, sublicense, sell, lease, host, publish, or otherwise make the Software or its proprietary compiled components available to third parties except with prior written authorization from ThousandML.

This restriction does not prevent distribution of model artifacts produced using the Software, subject to Section 8 and any applicable third-party model licenses. Any recipient who runs those artifacts with CQ runtime components for Commercial Use must hold its own Subscription.

8. MODELS, ADAPTERS, AND OUTPUTS

ThousandML does not claim ownership of your training data, your fine-tuned adapters, or the outputs of your models.

Model weights, adapters, and artifacts produced using the Software are not transferred to ThousandML merely because the Software was used to create them. Running them with CQ runtime components remains subject to Sections 2 and 3.

You are responsible for ensuring that you have all rights necessary to use, modify, quantize, fine-tune, distribute, or publish the underlying model.

Distribution and use of a CQ-quantized model or adapter remain subject to:

(a) the license of the original model;

(b) applicable laws and regulations;

(c) this license and any applicable Subscription or enterprise agreement; and

(d) any third-party licenses associated with the model or its accompanying software.

This license does not grant rights to any third-party model, dataset, trademark, or other intellectual property.

9. AUTHORIZED RUNTIME

CQ-quantized models and adapters trained on them depend on proprietary CQ runtime components.

You may not modify a CQ model, runtime, integration, or related software for the purpose of bypassing authorization or enabling unauthorized execution of proprietary CQ functionality.

10. NO TRANSFER OF INTELLECTUAL PROPERTY

The Software is licensed, not sold.

ThousandML and its licensors retain all right, title, and interest in and to the Software and proprietary CQ technology, including all associated intellectual property rights.

Except for the limited rights expressly granted by this license, no rights are granted by implication, estoppel, or otherwise.

11. THIRD-PARTY SOFTWARE

The Software may depend upon or interact with third-party software, including open-source software.

Third-party components remain subject to their respective licenses. Nothing in this license is intended to restrict rights granted under an applicable third-party license.

Where a third-party license conflicts with this license with respect to that third-party component, the third-party license controls for that component.

12. RESTRICTIONS

You may not use the Software:

(a) in violation of applicable law;

(b) to infringe or misappropriate the intellectual property rights of another party;

(c) to circumvent technological restrictions imposed by the Software or an underlying model provider; or

(d) in a manner that falsely represents an unauthorized product or implementation as being produced, approved, certified, or endorsed by ThousandML.

13. UPDATES AND ACCESS

ThousandML may provide updates, replacements, compatibility releases, or new versions of the Software.

Access to particular proprietary components or services may depend on authorization, licensing status, software version, or other eligibility requirements.

Except as stated in an active Subscription or separate written agreement, nothing in this license requires ThousandML to provide continued access, maintenance, support, updates, compatibility, or hosted services.

14. DISCLAIMER OF WARRANTY

THE SOFTWARE IS PROVIDED "AS IS" AND "AS AVAILABLE," WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE, AND NON-INFRINGEMENT.

THOUSANDML DOES NOT WARRANT THAT THE SOFTWARE WILL BE ERROR-FREE, UNINTERRUPTED, SECURE, OR COMPATIBLE WITH EVERY MODEL, HARDWARE CONFIGURATION, CUDA VERSION, DRIVER, OPERATING SYSTEM, OR THIRD-PARTY SOFTWARE VERSION.

15. LIMITATION OF LIABILITY

TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THOUSANDML AND ITS CONTRIBUTORS, AFFILIATES, LICENSORS, AND SUPPLIERS SHALL NOT BE LIABLE FOR ANY INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, CONSEQUENTIAL, OR PUNITIVE DAMAGES, OR FOR ANY LOSS OF DATA, MODELS, REVENUE, PROFITS, BUSINESS, OR COMPUTING RESOURCES, ARISING FROM OR RELATED TO THE SOFTWARE OR ITS USE.

TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THOUSANDML'S TOTAL LIABILITY ARISING FROM OR RELATED TO THE SOFTWARE SHALL NOT EXCEED THE AMOUNT PAID BY YOU TO THOUSANDML FOR THE SOFTWARE DURING THE TWELVE MONTHS PRECEDING THE EVENT GIVING RISE TO THE CLAIM.

IF YOU PAID NOTHING FOR THE SOFTWARE, THOUSANDML'S TOTAL LIABILITY SHALL NOT EXCEED USD $100.

16. TERMINATION

Your rights under this license terminate automatically if you materially violate its terms.

Upon termination, you must cease using and distributing the Software and delete proprietary CQ binaries in your possession, except to the extent retention is required by applicable law.

Sections concerning intellectual property, restrictions, warranty disclaimers, limitations of liability, and other provisions that by their nature should survive termination shall survive.

17. SEPARATE COMMERCIAL TERMS

ThousandML may offer enterprise agreements, volume pricing, support agreements, or other arrangements under separate terms.

Where you and ThousandML have entered into a separate written agreement governing the Software, that agreement controls to the extent of any conflict with this license.

18. CHANGES TO THIS LICENSE

ThousandML may publish new versions of this license. A new version applies to Software releases distributed under it. Releases you already obtained remain governed by the license version distributed with them, except that the Early Access Period ends as described in Section 4.

19. GOVERNING LAW

This license is governed by the laws of the State of California, without regard to its conflict-of-laws rules. Any dispute arising from this license shall be brought in the state or federal courts located in California, unless otherwise required by applicable law.

20. ACCEPTANCE

By downloading, installing, copying, accessing, or using the Software, you agree to the terms of this license.

If you do not agree to these terms, do not use the Software.

Commercial licensing: contact@thousandml.com

Copyright (c) 2026 Husam Elfadil, doing business as ThousandML. All rights reserved.
