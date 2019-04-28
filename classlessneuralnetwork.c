# include <stdbool.h>
# include <math.h>
# include <stdio.h>
# include <stdlib.h>
# include <malloc.h>
# include <string.h>

const double E = 2.718281828459;

float sig(float x) {
    float denominator = 1 + (float)pow(E, -x);
    return 1 / denominator;
}

int len(int *array) {
    return sizeof(array)/sizeof(array[0]);
}

float randFloat() {
    return (float)rand() / (float)RAND_MAX;
}

// +++ Network +++ //
int networkSize(int *widths) {
    int totalSize = len(widths) + 2;
    int nodeSize = 0;
    for (int i = 0; i < len(widths); i++) {
        nodeSize = 2 + widths[i - 1];
        if (i == 0) nodeSize = 1;
        totalSize += widths[i] * nodeSize;
    }
    return totalSize;
}

float * newNetwork(int *widths) {
    float *network = (float *)malloc(sizeof(float) * networkSize(widths));
    network[0] = len(widths);
    for (int i = 0; i < len(widths); i++) { network[1 + i] = widths[i]; }
    for (int i = 0; i < widths[0]; i++) {
        network[1 + len(widths) + i] = 0;
    }
    int pushIndex = 1 + len(widths) + widths[0];
    for (int i = 1; i < len(widths); i++) {
        for (int j = 0; j < widths[i]; j++) {
            network[pushIndex] = 0;
            pushIndex++;
            network[pushIndex] = 0;
            pushIndex++;
            for (int k = 0; k < widths[i - 1]; k++) {
                network[pushIndex] = 0;
                pushIndex++;
            }
        }
    }
    return network;
}

int nodeSize(float *network, int layer) {
    if (layer == 0) return 1;
    return 1 + network[2 + layer - 1];
}

int layerSize(float *network, int layer) {
    if (layer == 0) return network[1];
    return network[1 + layer] * nodeSize(network, layer);
}

int nodeAt(float *network, int layer, int node) {
    int nodeAt = network[0] + 1;
    for (int i = 0; i < layer; i++) {
        nodeAt += layerSize(network, layer);
    }
    for (int i = 0; i < node; i++) {
        nodeAt += nodeSize(network, layer);
    }
    return nodeAt;
}

float * createInputLayer(float *network, int *inputs) {
    for (int i = 0; i < network[1]; i++) {
        network[2 + (int)network[0] + i] = inputs[i];
    }
    return network;
}

// +++ Node +++ //
float * mutateNode(float *network, int layer, int node, float mutationRate) {
    for (int i = 1; i < nodeSize(network, layer); i++) {
        if (random() < mutationRate) {
            network[nodeAt(network, layer, node) + i] += (random() * 0.2) - 0.1;
            if (i > 1) {
                if (network[nodeAt(network, layer, node) + i] > 1) network[nodeAt(network, layer, node)] = 1;
                if (network[nodeAt(network, layer, node) + i] < -1) network[nodeAt(network, layer, node)] = -1;
            }
        }
    }
    return network;
}

float * setOutput(float *network, int layer, int node) {
    float output = network[nodeAt(network, layer, node) + 1];
    for (int i = 0; i < network[1 + layer - 1]; i++) {
        output += network[nodeAt(network, layer - 1, i)] * network[nodeAt(network, layer, node) + 2 + i];
    }
    network[nodeAt(network, layer, node)] = sig(output);
    return network;
}
// --- Node --- //

float * mutate(float *network, float mutationRate) {
    for (int layer = 1; layer < network[0]; layer++) {
        for (int node = 0; node < network[1 + layer]; node++) {
            network = mutateNode(network, layer, node, mutationRate);
        }
    }
    return network;
}

float * runNetwork(float *network) {
    for (int layer = 1; layer < network[0]; layer++) {
        for (int node = 0; node < network[1 + layer]; node++) {
            network = setOutput(network, layer, node);
        }
    }
    return network;
}
// --- Network --- //

int questionNumber = -1;
int question[2500];
int answer;

void getNewQuestion(int *fileLines, int lineSize) {
    questionNumber = (questionNumber + 1) % 6144;
    for (int i = 1; i < 2501; i++) {
        question[i - 1] = fileLines[questionNumber * lineSize + i];
    }
    answer = fileLines[questionNumber * lineSize + 0];
}

int largestIndex(int *array) {
    int currentLargest = array[0];
    int currentIndex = 0;
    for (int i = 1; i < len(array); i++) {
        if (array[i] > currentLargest) {
            currentLargest = array[i];
            currentIndex = i;
        }
    }
    return currentIndex;
}

float * removeElementFromArray(int *array, int index) {
    float *newArray = (float *)malloc(sizeof(float) * (len(array) - 1));    
    for (int i = 0; i < len(array); i++) {
        newArray[i] = array[i];
        if (i > index) newArray[i - 1] = array[i];
    }
    return newArray;
}

int main() {
    FILE *fp;
    printf("Reading file...\n");
    fp = fopen("tests.csv", "r");
    char ch;
    int lineSize;
    while ((ch = fgetc(fp)) != EOF) {
        if (ch == '\n') {
            lineSize = (ftell(fp) / sizeof(char)) / 2;
            break;
        }
    }
    fseek(fp, 0L, SEEK_END);
    int *fileLines = malloc(sizeof(int) * (ftell(fp) / sizeof(char)) / 2);
    rewind(fp);
    int index = 0;
    int fileIndex = 0;
    while ((ch = fgetc(fp)) != EOF) {
        if (ch != '\n') {
            if (ch != ',') {
                fileLines[fileIndex * lineSize + index] = (int)ch;
                index++;
            }
        }
        else {
            fileIndex++;
            index = 0;
        }
    }
    fclose(fp);
    printf("Done\n");
    printf("Training networks...\n");

    while (1 == 1) {
        int networkLayers[2] = {2500, 1};
        int size = networkSize(networkLayers);
        float *generation = malloc(sizeof(float) * 200 * size);
        for (int i = 0; i < 200; i++) {
            memcpy(generation + i * size, newNetwork(networkLayers), size * sizeof(float));
        }
        int highestFitness;
        int *fitness = malloc(sizeof(int) * 200);
        for (int j = 0; j < 100; j++) {
            for (int i = 0; i < 200; i++) fitness[i] = 0;
            for (int t = 0; t < 6144; t++) {
                getNewQuestion(fileLines, lineSize);
                for (int i = 0; i < 200; i++) {
                    memcpy(generation + i * size, runNetwork(createInputLayer(generation + i * size, question)), size * sizeof(float));
                    if (round(generation[i * size + nodeAt(generation + i * size, 1, 0)]) == answer) fitness[i]++;
                }
            }
            highestFitness = fitness[largestIndex(fitness)];
            float *best = malloc(sizeof(float) * 5 * size);
            for (int i = 0; i < 5; i++) {
                memcpy(best + i * size, generation + largestIndex(fitness) * size, size * sizeof(float));
                for (int i = 0; i < len(fitness); i++) {
                    fitness[i] = removeElementFromArray(fitness, largestIndex(fitness))[i];
                }
            }
            memcpy(generation, best, 5 * size * sizeof(float));
            for (int i = 5; i < 200; i++) {
                memcpy(generation + i * size, mutate(generation + (i % 5) * size, -highestFitness / 3100 + 2), size * sizeof(float));
            }
            printf("Error rate (percent): %f\n", (6144 - highestFitness) / 61.44);
        }
        if (6144 - highestFitness <= 5) {
            fp = fopen("bestnetworks.txt", "a");
            fprintf(fp, "%d", 6144 - highestFitness);
            fputc(',', fp);
            int *networkWidths = malloc(sizeof(int) * generation[largestIndex(fitness) * size + 0]);
            for (int i = 0; i < generation[largestIndex(fitness) * size + 0]; i++) {
                networkWidths[i] = generation[largestIndex(fitness) * size + i + 1];
            }
            char *networkStr = malloc(sizeof(char) * networkSize(networkWidths));
            for (int i = 0; i < networkSize(networkWidths); i++) {
                fprintf(fp, "%f", generation[largestIndex(fitness) * size + i]);
                if (i + 1 < networkSize(networkWidths)) fputc(',', fp);
            }
            fputc('\n', fp);
            fclose(fp);
        }
    }
    return 0;
}