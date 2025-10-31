# Potato Leaf Disease Classification Model

## Project Overview

Potato leaf diseases represent a significant threat to global food security, encompassing fungal, bacterial, and viral infections such as early blight, late blight, and leaf roll virus. These diseases can cause spots, wilting, yellowing, and substantially reduced crop yields, leading to economic losses for farmers worldwide.

This project implements a deep learning solution using PyTorch to automatically classify potato leaf diseases from images, enabling early detection and intervention to protect crops.

## Introduction

Potatoes are one of the world's most important staple crops, yet they are vulnerable to numerous diseases that can devastate entire harvests. Traditional disease identification requires expert knowledge and manual inspection, which is time-consuming and often occurs too late for effective treatment.

### Types of Potato Leaf Diseases

#### Fungal Diseases
The most prevalent category, often spreading in humid conditions:
- **Early Blight** (*Alternaria solani*): Dark concentric spots on older leaves, leading to defoliation
- **Late Blight** (*Phytophthora infestans*): Rapidly spreads in cool, wet weather; causes water-soaked lesions and leaf decay
- **Black Dot** (*Colletotrichum coccodes*): Small black specks on leaves and stems
- **Brown Spot / Black Pit** (*Alternaria alternata*): Brown lesions that can merge and kill leaf tissue
- **Cercospora Leaf Blotch** (*Mycovellosiella concors*): Irregular brown patches with yellow halos

#### Bacterial Diseases
Often entering through wounds or insect damage:
- **Bacterial Wilt / Brown Rot** (*Ralstonia solanacearum*): Causes wilting and yellowing of leaves
- **Blackleg** (*Pectobacterium carotovorum*): Leads to blackened stems and leaf collapse
- **Ring Rot** (*Clavibacter michiganensis*): Yellowing and curling of leaves, often with internal tuber damage

#### Viral Diseases
Spread by insects like aphids and leafhoppers:
- **Potato Leaf Roll Virus (PLRV)**: Leaves curl upward and become leathery
- **Potato Virus Y (PVY)**: Mottling and mosaic patterns on leaves
- **Tomato Spotted Wilt Virus (TSWV)**: Can affect potatoes, causing necrotic spots and stunted growth

####  Pest-Related Disorders
Some pests cause symptoms that mimic disease:
- **Aphids**: Can transmit viruses and cause leaf curling
- **Spider Mites**: Lead to stippling and bronzing of leaves

## Model Performance

Our PyTorch-based deep learning model demonstrates excellent performance in classifying various potato and tomato leaf diseases:

### Overall Metrics
- **Training Accuracy**: 91%
- **Validation Accuracy**: 94%
- **Overall Precision**: 0.94
- **Overall Recall**: 0.94
- **Overall F1-Score**: 0.94

### Per-Class Performance

| Disease Class | Precision | Recall | F1-Score | Support |
|--------------|-----------|---------|----------|---------|
| Pepper Bell Bacterial Spot | 0.82 | 0.99 | 0.90 | 194 |
| Pepper Bell Healthy | 0.95 | 1.00 | 0.97 | 314 |
| Potato Early Blight | 0.96 | 0.98 | 0.97 | 177 |
| Potato Late Blight | 0.97 | 0.83 | 0.89 | 201 |
| Potato Healthy | 1.00 | 0.85 | 0.92 | 26 |
| Tomato Bacterial Spot | 0.99 | 0.97 | 0.98 | 460 |
| Tomato Early Blight | 0.90 | 0.82 | 0.86 | 208 |
| Tomato Late Blight | 0.86 | 0.92 | 0.89 | 352 |
| Tomato Leaf Mold | 0.88 | 0.95 | 0.91 | 186 |
| Tomato Septoria Leaf Spot | 0.99 | 0.84 | 0.91 | 348 |
| Tomato Spider Mites (Two-Spotted) | 0.96 | 0.93 | 0.94 | 334 |
| Tomato Target Spot | 0.92 | 0.93 | 0.93 | 300 |
| Tomato Yellow Leaf Curl Virus | 0.97 | 0.99 | 0.98 | 616 |
| Tomato Mosaic Virus | 0.95 | 1.00 | 0.98 | 79 |
| Tomato Healthy | 0.99 | 0.99 | 0.99 | 333 |

The model shows particularly strong performance on:
- **Potato Early Blight**: 96% precision, 98% recall
- **Tomato Yellow Leaf Curl Virus**: 97% precision, 99% recall
- **Tomato Bacterial Spot**: 99% precision, 97% recall
- **Healthy leaves**: Consistently high accuracy across plant types

## Key Features

-  Multi-class classification across 15 different plant health conditions
-  High accuracy detection of potato-specific diseases (Early Blight, Late Blight)
-  Extended capability to classify pepper and tomato diseases
-  Balanced performance across precision and recall metrics
-  PyTorch implementation for efficient training and inference

## Applications

This model can be deployed in:
- Mobile applications for farmers to diagnose diseases in the field
- Automated monitoring systems in commercial greenhouses
- Agricultural extension services for rapid disease assessment
- Educational tools for agricultural training programs
