#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* ===========*/
/*   STRUCTS  */
/* ===========*/

typedef enum {INPUT_SPIKE=0,NEURON_SPIKE=1} event_type;

/* ELEMENTOS DE ENTRADA */

typedef struct config_{
    int n_neu;
    int n_inputs;
    float threshold;
    float v_rest;
    float tau;
    float refractory_time;
    float syn_delay;
    float coda;
    float max_sim_time;
} Config;

typedef struct inputs_raw{
    int    len;
    int*   idx;
    float* times;
} InputsRaw;

typedef struct synapses_raw{
    int len;
    int* pre;
    int* post;
    float* w;
} SynapsesRaw;

/* ELEMENTOS INTERNOS */

typedef struct node{
    void*        data;
    struct node* next;
} Node;

typedef struct synapses{
    int post;
    float w;
} Synapses;

typedef struct spike{
    event_type type;
    int        pre;
    float      time;
} Spike;

typedef struct voltage{
    float volt;
    float time;
} Voltage;

/* ELEMENTOS DE SALIDA */
typedef struct voltage_marker{
    int neu;
    float time;
    float voltage;
} VoltageMarker;

typedef struct spike_marker{
    int neu;
    float time;
} SpikeMarker;

/* ===========*/
/*   HEADERS  */
/* ===========*/
Node* addNode(void* data, Node* list);                                    /* Manejo de listas */
void freeList(Node* list);

Node* inputsRaw_to_spikeList(InputsRaw inputs_raw);                       /* Spikes */
void debug_spikes(Node* list_spikes);
void insertSpike(int pre, float time, Node* list_spikes);

Node** synapsesRaw_to_synapsesArray(SynapsesRaw synapses_raw, int n_pre); /* Synapses */
void debugSynapses(Node** synapses, int n_pre);
void freeSynapses(Node** synapses, int n_pre);

VoltageMarker* createVoltageMaker(int neu, float time, float voltage);    /* Rec */
SpikeMarker* createSpikeMarker(int neu, float time);

/* ===========*/
/*   SIMULAR  */
/* ===========*/

/*   LIF EQUATION   */
float LIF_eq(float v_rest, float v0, float time, float t_init, float tau){
    return v_rest + (v0-v_rest)*exp( -(time - t_init)/tau );
}

void simulate(Config config, InputsRaw inputs, SynapsesRaw synapses_in, SynapsesRaw synapses_re, Node** voltages, Node** spikes){
    
    /* =================== */
    /* ESTRUCTURA DE LA RED*/
    Node* list_spikes = inputsRaw_to_spikeList(inputs);
    Node* list_spikes_head = list_spikes;
    Node** syn_in = synapsesRaw_to_synapsesArray(synapses_in, config.n_inputs);
    Node** syn_re = synapsesRaw_to_synapsesArray(synapses_re, config.n_neu);


    /* ============================ */
    /* INICIALIZAR ESTADOS INTERNOS */
    Voltage neu_state[config.n_neu];           /* Voltages internos*/
    for(int i=0 ; i<config.n_neu ; i++){
        neu_state[i].time = 0;
        neu_state[i].volt = config.v_rest;
    }

    float refractory[config.n_neu];             /* Refractory */
    for(int i=0 ; i<config.n_neu ; i++){
        refractory[i] = -config.refractory_time;
    }

    /* ======== */
    /* REC STUFF*/
    Node* voltage_rec = NULL;
    for(int i=0 ; i<config.n_neu ; i++){        /* Voltajes en t=0 */
        VoltageMarker* marker = createVoltageMaker(i,0,config.v_rest);
        voltage_rec = addNode(marker,voltage_rec);
    }
    Node* spike_rec = NULL;

    /* ========= */
    /* MAIN LOOP */
    float current_time = 0;
    while(list_spikes){             // Recorrer todos los eventos
        Spike* spike = (Spike*)list_spikes->data;

        current_time = spike->time;                // No superar max time
        if(current_time >= config.max_sim_time){   // to prevent infinite loops
            break;
        }

        Node* syn_list;                     // Cual sinapses debo usar (input o reservoir)
        Synapses* syn;
        if (spike->type==INPUT_SPIKE) syn_list= syn_in[spike->pre];
        else                          syn_list= syn_re[spike->pre];

        Node* current;
        /* LEAKY */
        current = syn_list;
        while(current){                     // Recorrer todas las sinapsis del evento
            syn = (Synapses*)current->data;

            // check refractory
            if(current_time - refractory[syn->post] >= config.refractory_time){
                // leak
                float v0     = neu_state[syn->post].volt;
                float t_init = neu_state[syn->post].time;
                float new_v  = LIF_eq(config.v_rest, v0, current_time, t_init, config.tau);
                neu_state[syn->post].volt = new_v;
                neu_state[syn->post].time = current_time;
            }

            /* next */
            current = current->next;
        }

        /* INTEGRATE */
        current = syn_list;
        Node* to_fire_list = NULL;
        while (current){
            syn = (Synapses*)current->data;

            // check refractory
            if(current_time - refractory[syn->post] >= config.refractory_time){
                // sumar voltages
                neu_state[syn->post].volt += syn->w;

                // check if spike
                if(neu_state[syn->post].volt >= config.threshold){
                    int* to_fire = malloc(sizeof(int));
                    *to_fire = syn->post;
                    to_fire_list = addNode(to_fire, to_fire_list);
                }
            }

            /* next */
            current = current->next;
        }

        /* REC */
        current = syn_list;
        while(current){
            syn = (Synapses*)current->data;

            VoltageMarker* marker = createVoltageMaker(syn->post, current_time, neu_state[syn->post].volt);
            voltage_rec = addNode(marker, voltage_rec);

            /* next */
            current = current->next;
        }

        /* AND FIRE */
        int* neu;
        Node* to_fire_list_head = to_fire_list;
        while(to_fire_list){
            neu = (int*)to_fire_list->data;

            int spike_neu    = *neu;
            float spike_time = current_time+config.syn_delay;

            /* Gestionar reset y evento para las neu post */
            neu_state[spike_neu].volt = config.v_rest;
            neu_state[spike_neu].time = spike_time;

            /* evento para las neu post */
            insertSpike(spike_neu, spike_time, list_spikes);

            /* Rec voltajes */
            VoltageMarker* v_marker = createVoltageMaker(spike_neu, spike_time, config.v_rest);
            voltage_rec             = addNode(v_marker,voltage_rec);

            /* Rec spikes */
            SpikeMarker* s_marker = createSpikeMarker(spike_neu, spike_time);
            spike_rec             = addNode(s_marker, spike_rec);

            /* Set Refractory */
            refractory[spike_neu] = spike_time;

            /* next */
            to_fire_list = to_fire_list->next;
        }
        freeList(to_fire_list_head);

        list_spikes = list_spikes->next;
    }

    /* CODA: Final Voltages */
    for(int neu=0 ; neu<config.n_neu ; neu++){
        float v0              = neu_state[neu].volt;
        float t_init          = neu_state[neu].time;
        float t_end           = current_time+config.coda;
        float new_voltage     = LIF_eq(config.v_rest, v0, t_end, t_init, config.tau );
        VoltageMarker* marker = createVoltageMaker(neu,t_end,new_voltage);
        voltage_rec           = addNode(marker,voltage_rec);
    }


    /* SAVE DATA */
    *voltages = voltage_rec;
    *spikes   = spike_rec;

    /* FREE POINTERS */
    freeList(list_spikes_head);
    freeSynapses(syn_in, config.n_inputs);
    freeSynapses(syn_re, config.n_neu);
}

/* ===========*/
/*  FUNCIONES */
/* ===========*/

/* Manejo de listas */
Node* addNode(void* data, Node* list){
    Node* new_node = malloc(sizeof(Node));
    new_node->data = data;
    new_node->next = list;
    return new_node;
}

void freeList(Node* list){
    Node* next;
    while(list){
        next = list->next;
        free(list->data);
        free(list);
        list = next;
    }
}

/* Spikes */
Node* inputsRaw_to_spikeList(InputsRaw inputs){
    Node*  spike_list    = NULL;
    Spike* new_spike;
    
    for(int i=0 ; i<inputs.len ; i++){
        new_spike = malloc(sizeof(Spike));
        new_spike->pre  = inputs.idx[i];
        new_spike->time = inputs.times[i];
        new_spike->type = INPUT_SPIKE;

        spike_list = addNode(new_spike, spike_list);
    }
    return spike_list;
}

void debug_spikes(Node* list_spikes){
    Spike* current;
    while(list_spikes){
        current = (Spike*) list_spikes->data;
        printf("idx:%i time:%.3f\n",current->pre, current->time);
        list_spikes = list_spikes->next;
    }
}

void insertSpike(int pre, float time, Node* list_spikes){
    Spike* spike = malloc(sizeof(Spike));
    spike->pre   = pre;
    spike->time  = time;
    spike->type  = NEURON_SPIKE;    

    Node* to_insert = malloc(sizeof(Node));
    to_insert->data = spike;
    to_insert->next = NULL;
    
    Node* next;
    while(list_spikes){
        next = list_spikes->next;

        if(!next){
            list_spikes->next = to_insert;
            break;
        } 
        else{
            Spike* next_spike = next->data;
            if(time < next_spike->time){
                to_insert->next  = next;
                list_spikes->next = to_insert;
                break;
            }
        }
        list_spikes = next;
    }
}

/* Synapses */
Node** synapsesRaw_to_synapsesArray(SynapsesRaw synapses_raw, int n_pre){
    Node** synapses = malloc( sizeof(Node*)*n_pre );
    for(int i=0 ; i<n_pre ; i++){
        synapses[i] = NULL;
    }
    int pre, post;
    float w;
    for(int idx=0 ; idx<synapses_raw.len ; idx++){
        pre  = synapses_raw.pre[idx];
        
        Synapses* syn = malloc(sizeof(Synapses));
        syn->post = synapses_raw.post[idx];
        syn->w    = synapses_raw.w[idx];

        synapses[pre] = addNode(syn,synapses[pre]);

    }
    return synapses;
}

void debugSynapses(Node** synapses, int n_pre){
    Node* current;
    Synapses* syn;
    for(int i=0 ; i<n_pre ; i++){
        current = synapses[i];
        printf("Neu %i: ",i);
        while(current){
            syn = (Synapses*)current->data;
            printf("(%i, %.f) ",syn->post, syn->w);
            current = current->next;
        }
        printf("\n");
    }
}

void freeSynapses(Node** synapses, int n_pre){
    for(int i=0 ; i<n_pre ; i++){
        freeList( synapses[i] );
        synapses[i] = NULL;
    }
}

/*  Rec */
VoltageMarker* createVoltageMaker(int neu, float time, float voltage){
    VoltageMarker* new_v = malloc(sizeof(VoltageMarker));
    new_v->neu      = neu;
    new_v->time     = time;
    new_v->voltage  = voltage;
    return new_v;
}

SpikeMarker* createSpikeMarker(int neu, float time){
    SpikeMarker* new_s = malloc(sizeof(SpikeMarker));
    new_s->neu  = neu;
    new_s->time = time;
    return new_s;
}