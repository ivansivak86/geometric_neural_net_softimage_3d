/*******************************************************************************
 * NeuralGraph v0.3 for SOFTIMAGE|3D 4.0
 *
 * Learned graph diffusion, animated endpoint gradients, message-flux motion,
 * and a second PDE-style diffusion playback phase.
 *
 * Visual phases baked into the native SOFTIMAGE|3D timeline:
 *   1. Training: edge couplings are optimized against a smooth target field.
 *   2. Hold: the final learned state remains visible for a short pause.
 *   3. Diffusion playback: learned |weights| become graph conductivities and
 *      a fresh bipolar source diffuses through the frozen graph.
 *
 * Target: Microsoft Visual C++ 6.0 / SDK 4.0 / SAAPHIRE.
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <SAA.h>

#ifndef TRUE
#define TRUE 1
#endif
#ifndef FALSE
#define FALSE 0
#endif

/* Dialog items. */
#define NG_ITEM_DONE                    1
#define NG_ITEM_NODE_COUNT             14
#define NG_ITEM_NEIGHBOURS             16
#define NG_ITEM_PROP_STEPS             18
#define NG_ITEM_EPOCHS                 20
#define NG_ITEM_LEARNING_RATE          22
#define NG_ITEM_LAYOUT_RADIUS          24
#define NG_ITEM_NODE_RADIUS            26
#define NG_ITEM_EDGE_RADIUS            28
#define NG_ITEM_WEIGHT_SCALE           30
#define NG_ITEM_RANDOM_SEED            32
#define NG_ITEM_FRAMES_PER_EPOCH       34
#define NG_ITEM_LAYOUT_SPHERE          36
#define NG_ITEM_LAYOUT_RING            37
#define NG_ITEM_SIGNED_WEIGHTS         38
#define NG_ITEM_MOTION_RADIAL          39
#define NG_ITEM_MOTION_FLUX            40
#define NG_ITEM_MOTION_STATIC          41
#define NG_ITEM_MOTION_AMOUNT          43
#define NG_ITEM_NODE_SCALE_GAIN        45
#define NG_ITEM_EDGE_THICKNESS_GAIN    47
#define NG_ITEM_GRADIENT_SEGMENTS      49
#define NG_ITEM_PULSE_COUNT            51
#define NG_ITEM_GAP_FRAMES             53
#define NG_ITEM_DIFFUSION_STEPS        55
#define NG_ITEM_FRAMES_PER_DIFFUSION   57
#define NG_ITEM_CREATE                 62

/* Limits. */
#define NG_MAX_NODES                   32
#define NG_MAX_EDGES                  192
#define NG_MAX_EPOCHS                 120
#define NG_MAX_DIFFUSION_STEPS         48
#define NG_MAX_GRADIENT_SEGMENTS        6
#define NG_MAX_PULSES                  12
#define NG_EDGE_SIDES                   6
#define NG_EPSILON                 1.0e-8
#define NG_PI                     3.14159265358979323846
#define NG_GOLDEN_ANGLE           2.39996322972865332223
#define NG_STATUS_BUFFER             256

#define NG_PHASE_TRAINING               0
#define NG_PHASE_HOLD                   1
#define NG_PHASE_DIFFUSION              2

#define NG_MOTION_STATIC                0
#define NG_MOTION_RADIAL                1
#define NG_MOTION_FLUX                  2

typedef struct { double x, y, z; } NG_Vector3;
typedef struct { float r, g, b; } NG_Color;

typedef struct
{
   int a;
   int b;
   float weight;
   float distanceSquared;
} NG_Edge;

typedef struct
{
   int nodeCount;
   int neighbours;
   int propagationSteps;
   int epochs;
   float learningRate;
   float layoutRadius;
   float nodeRadius;
   float edgeRadius;
   float weightScale;
   int randomSeed;
   int framesPerEpoch;
   SAA_Boolean sphericalLayout;
   SAA_Boolean signedWeights;
   int motionMode;
   float motionAmount;
   float nodeScaleGain;
   float edgeThicknessGain;
   int gradientSegments;
   int pulseCount;
   int gapFrames;
   int diffusionSteps;
   int framesPerDiffusion;
} NG_Parameters;

typedef struct { unsigned long state; } NG_Random;

typedef struct
{
   int edgeCount;
   NG_Vector3 positions[NG_MAX_NODES];
   NG_Edge edges[NG_MAX_EDGES];
   float sourceField[NG_MAX_NODES];
   float targetField[NG_MAX_NODES];
   float *epochNodeValues;
   float *epochEdgeWeights;
   float *epochLosses;
} NG_TrainingResult;

typedef struct
{
   int count;
   SAA_DVector *vertices;
   NG_Vector3 *unitDirections;
} NG_NodeTemplate;

typedef struct
{
   int keyCount;
   int *frames;
   float *times;
   int *phases;
   int *phaseSteps;
   float *nodeValues;
   float *edgeWeights;
} NG_AnimationData;

typedef struct
{
   float maximumNodeValue;
   float maximumEdgeWeight;
   float maximumFlow;
} NG_VisualMetrics;

typedef struct
{
   SAA_Elem *items;
   int count;
   int capacity;
} NG_MaterialList;

static SAA_Scene g_scene;
static unsigned long g_graphSerial = 0;
static char g_statusGenerating[] =
   "NeuralGraph v0.3: training, diffusion and animation baking...";
static char g_statusComplete[NG_STATUS_BUFFER];
static char g_statusFailed[] =
   "NeuralGraph v0.3: generation failed; partial geometry removed.";

/* Core callbacks. */
static SI_Error ngCreateCallback(const SAA_CustomContext context, int item, void *userData);
static SI_Error ngReadParameters(const SAA_CustomContext context, NG_Parameters *parameters);
static SI_Error ngGenerateAnimation(const SAA_CustomContext context, const NG_Parameters *parameters);

/* Graph and learning. */
static void ngBuildLayout(const NG_Parameters *, NG_Random *, NG_Vector3 *);
static int ngBuildEdges(const NG_Parameters *, NG_Random *, const NG_Vector3 *, NG_Edge *);
static void ngBuildSourceAndTargetFields(const NG_Parameters *, const NG_Vector3 *, float *, float *);
static void ngForwardPropagate(const NG_Parameters *, const NG_Edge *, int, const float *, float *);
static void ngTrainGraph(const NG_Parameters *, NG_TrainingResult *);
static float ngComputeLoss(int, const float *, const float *);
static void ngOptimizeWeights(const NG_Parameters *, NG_TrainingResult *, int);

/* Timeline and diffusion. */
static SI_Error ngBuildAnimationData(const NG_Parameters *, const NG_TrainingResult *, int, NG_AnimationData *);
static void ngSimulateDiffusion(const NG_Parameters *, const NG_TrainingResult *, float *);
static void ngFreeAnimationData(NG_AnimationData *);
static void ngComputeVisualMetrics(const NG_Parameters *, const NG_TrainingResult *, const NG_AnimationData *, NG_VisualMetrics *);

/* Geometry and motion. */
static void ngComputeNodeFlux(const NG_Parameters *, const NG_TrainingResult *, const float *, const float *, int, NG_Vector3 *);
static void ngBuildAnimatedNodePose(const NG_Parameters *, const NG_TrainingResult *, const float *, const float *, int, NG_Vector3 *, double *);
static double ngComputeEdgeRadius(const NG_Parameters *, float, float);
static void ngBuildAnimatedEdgeVertices(NG_Vector3, NG_Vector3, double, SAA_DVector *);
static void ngComputeShortenedEdge(NG_Vector3, NG_Vector3, double, double, NG_Vector3 *, NG_Vector3 *);
static void ngComputeSegmentEndpoints(NG_Vector3, NG_Vector3, int, int, NG_Vector3 *, NG_Vector3 *);
static SI_Error ngCreateEdgeMesh(const SAA_Scene *, NG_Vector3, NG_Vector3, double, SAA_Elem *);
static SI_Error ngPrepareMeshNormals(const SAA_Scene *, const SAA_Elem *);

/* Shape keys. */
static SI_Error ngGetTemplateFromSphere(const SAA_Scene *, const SAA_Elem *, NG_NodeTemplate *);
static void ngFreeNodeTemplate(NG_NodeTemplate *);
static void ngBuildNodeVertices(const NG_NodeTemplate *, NG_Vector3, double, SAA_DVector *);
static SI_Error ngBakeNodeShapes(const NG_Parameters *, const NG_TrainingResult *, const NG_AnimationData *, const NG_NodeTemplate *, const SAA_Elem *, int);
static SI_Error ngBakeEdgeSegmentShapes(const NG_Parameters *, const NG_TrainingResult *, const NG_AnimationData *, const NG_VisualMetrics *, const SAA_Elem *, int, int);

/* Flow pulses. */
static void ngBuildPulsePose(const NG_Parameters *, const NG_TrainingResult *, const NG_AnimationData *, const NG_VisualMetrics *, int, int, int, NG_Vector3 *, double *, float *);
static SI_Error ngBakePulseShapes(const NG_Parameters *, const NG_TrainingResult *, const NG_AnimationData *, const NG_VisualMetrics *, const NG_NodeTemplate *, const SAA_Elem *, int, int);
static int ngSelectStrongestEdges(const NG_Parameters *, const NG_TrainingResult *, int *);

/* Animated materials. */
static void ngActivationColor(float, float, NG_Color *);
static void ngWeightColor(float, float, NG_Color *);
static void ngEdgeGradientColor(float, float, float, double, const NG_VisualMetrics *, NG_Color *);
static void ngPulseColor(int, float, float, NG_Color *);
static SI_Error ngCreateAnimatedMaterial(const SAA_Scene *, const char *, const NG_AnimationData *, const NG_Color *, SAA_Elem *);
static SI_Error ngAddFcurveKeys(const SAA_Scene *, const SAA_Elem *, int, const float *, const float *);
static SI_Error ngAssignGlobalMaterial(const SAA_Scene *, const SAA_Elem *, const SAA_Elem *);
static SI_Error ngMaterialListInit(NG_MaterialList *, int);
static SI_Error ngMaterialListAdd(NG_MaterialList *, const SAA_Elem *);
static void ngMaterialListRelease(const SAA_Scene *, NG_MaterialList *, SAA_Boolean);

/* Cleanup. */
static void ngCleanupPartialGraph(const SAA_Scene *, SAA_Boolean, SAA_Elem *, NG_MaterialList *);
static void ngFreeTrainingResult(NG_TrainingResult *);

/* Helpers. */
static int ngClampInt(int, int, int);
static float ngClampFloat(float, float, float);
static double ngAbs(double);
static double ngMax(double, double);
static double ngLength(NG_Vector3);
static double ngDistanceSquared(NG_Vector3, NG_Vector3);
static NG_Vector3 ngAdd(NG_Vector3, NG_Vector3);
static NG_Vector3 ngSubtract(NG_Vector3, NG_Vector3);
static NG_Vector3 ngScale(NG_Vector3, double);
static NG_Vector3 ngCross(NG_Vector3, NG_Vector3);
static NG_Vector3 ngNormalize(NG_Vector3);
static NG_Vector3 ngLerpVector(NG_Vector3, NG_Vector3, double);
static NG_Color ngLerpColor(NG_Color, NG_Color, double);
static NG_Color ngScaleColor(NG_Color, double);
static NG_Color ngAddColor(NG_Color, NG_Color);
static void ngClampColor(NG_Color *);
static unsigned long ngRandomNext(NG_Random *);
static float ngRandomUnit(NG_Random *);

#ifdef __cplusplus
extern "C" {
#endif

_CUS_EXTERN SI_Error dialogSetup(const SAA_CustomContext context)
{
   SI_Error result;

   result = SAA_sceneInit();
   if (result != SI_SUCCESS) return result;
   result = SAA_sceneGetCurrent(&g_scene);
   if (result != SI_SUCCESS) return result;

   SAA_dialogitemSetIntValue(context, NG_ITEM_NODE_COUNT, 12);
   SAA_dialogitemSetIntValue(context, NG_ITEM_NEIGHBOURS, 3);
   SAA_dialogitemSetIntValue(context, NG_ITEM_PROP_STEPS, 4);
   SAA_dialogitemSetIntValue(context, NG_ITEM_EPOCHS, 24);
   SAA_dialogitemSetFloatValue(context, NG_ITEM_LEARNING_RATE, 0.18f);
   SAA_dialogitemSetFloatValue(context, NG_ITEM_LAYOUT_RADIUS, 4.0f);
   SAA_dialogitemSetFloatValue(context, NG_ITEM_NODE_RADIUS, 0.30f);
   SAA_dialogitemSetFloatValue(context, NG_ITEM_EDGE_RADIUS, 0.055f);
   SAA_dialogitemSetFloatValue(context, NG_ITEM_WEIGHT_SCALE, 1.0f);
   SAA_dialogitemSetIntValue(context, NG_ITEM_RANDOM_SEED, 42);
   SAA_dialogitemSetIntValue(context, NG_ITEM_FRAMES_PER_EPOCH, 1);
   SAA_dialogitemSetStateValue(context, NG_ITEM_LAYOUT_SPHERE, TRUE);
   SAA_dialogitemSetStateValue(context, NG_ITEM_LAYOUT_RING, FALSE);
   SAA_dialogitemSetStateValue(context, NG_ITEM_SIGNED_WEIGHTS, TRUE);
   SAA_dialogitemSetStateValue(context, NG_ITEM_MOTION_RADIAL, FALSE);
   SAA_dialogitemSetStateValue(context, NG_ITEM_MOTION_FLUX, TRUE);
   SAA_dialogitemSetStateValue(context, NG_ITEM_MOTION_STATIC, FALSE);
   SAA_dialogitemSetFloatValue(context, NG_ITEM_MOTION_AMOUNT, 0.35f);
   SAA_dialogitemSetFloatValue(context, NG_ITEM_NODE_SCALE_GAIN, 1.10f);
   SAA_dialogitemSetFloatValue(context, NG_ITEM_EDGE_THICKNESS_GAIN, 1.90f);
   SAA_dialogitemSetIntValue(context, NG_ITEM_GRADIENT_SEGMENTS, 3);
   SAA_dialogitemSetIntValue(context, NG_ITEM_PULSE_COUNT, 6);
   SAA_dialogitemSetIntValue(context, NG_ITEM_GAP_FRAMES, 8);
   SAA_dialogitemSetIntValue(context, NG_ITEM_DIFFUSION_STEPS, 18);
   SAA_dialogitemSetIntValue(context, NG_ITEM_FRAMES_PER_DIFFUSION, 2);

   return SAA_dialogitemAddCallback(context, NG_ITEM_CREATE, ngCreateCallback, NULL);
}

#ifdef __cplusplus
}
#endif

static SI_Error ngCreateCallback(const SAA_CustomContext context, int item, void *userData)
{
   NG_Parameters parameters;
   SI_Error result;

   (void)userData;
   if (item != NG_ITEM_CREATE) return SI_ERR_BAD_ARGUMENT;

   result = ngReadParameters(context, &parameters);
   if (result != SI_SUCCESS) return result;

   SAA_statusBarSet(g_statusGenerating, SAA_MESSAGE_CODE);
   result = ngGenerateAnimation(context, &parameters);
   if (result == SI_SUCCESS)
      SAA_statusBarSet(g_statusComplete, SAA_MESSAGE_CODE);
   else
      SAA_statusBarSet(g_statusFailed, SAA_ERROR_CODE);
   return result;
}

static SI_Error ngReadParameters(const SAA_CustomContext context, NG_Parameters *parameters)
{
   SI_Error result;
   SAA_Boolean spherical, ring, radial, flux, staticMotion;

   if (parameters == NULL) return SI_ERR_BAD_ARGUMENT;

#define NG_GET_INT(item, field) \
   result = SAA_dialogitemGetIntValue(context, item, &parameters->field); \
   if (result != SI_SUCCESS) return result
#define NG_GET_FLOAT(item, field) \
   result = SAA_dialogitemGetFloatValue(context, item, &parameters->field); \
   if (result != SI_SUCCESS) return result

   NG_GET_INT(NG_ITEM_NODE_COUNT, nodeCount);
   NG_GET_INT(NG_ITEM_NEIGHBOURS, neighbours);
   NG_GET_INT(NG_ITEM_PROP_STEPS, propagationSteps);
   NG_GET_INT(NG_ITEM_EPOCHS, epochs);
   NG_GET_FLOAT(NG_ITEM_LEARNING_RATE, learningRate);
   NG_GET_FLOAT(NG_ITEM_LAYOUT_RADIUS, layoutRadius);
   NG_GET_FLOAT(NG_ITEM_NODE_RADIUS, nodeRadius);
   NG_GET_FLOAT(NG_ITEM_EDGE_RADIUS, edgeRadius);
   NG_GET_FLOAT(NG_ITEM_WEIGHT_SCALE, weightScale);
   NG_GET_INT(NG_ITEM_RANDOM_SEED, randomSeed);
   NG_GET_INT(NG_ITEM_FRAMES_PER_EPOCH, framesPerEpoch);
   NG_GET_FLOAT(NG_ITEM_MOTION_AMOUNT, motionAmount);
   NG_GET_FLOAT(NG_ITEM_NODE_SCALE_GAIN, nodeScaleGain);
   NG_GET_FLOAT(NG_ITEM_EDGE_THICKNESS_GAIN, edgeThicknessGain);
   NG_GET_INT(NG_ITEM_GRADIENT_SEGMENTS, gradientSegments);
   NG_GET_INT(NG_ITEM_PULSE_COUNT, pulseCount);
   NG_GET_INT(NG_ITEM_GAP_FRAMES, gapFrames);
   NG_GET_INT(NG_ITEM_DIFFUSION_STEPS, diffusionSteps);
   NG_GET_INT(NG_ITEM_FRAMES_PER_DIFFUSION, framesPerDiffusion);

#undef NG_GET_INT
#undef NG_GET_FLOAT

   result = SAA_dialogitemGetStateValue(context, NG_ITEM_LAYOUT_SPHERE, &spherical);
   if (result != SI_SUCCESS) return result;
   result = SAA_dialogitemGetStateValue(context, NG_ITEM_LAYOUT_RING, &ring);
   if (result != SI_SUCCESS) return result;
   result = SAA_dialogitemGetStateValue(context, NG_ITEM_SIGNED_WEIGHTS, &parameters->signedWeights);
   if (result != SI_SUCCESS) return result;
   result = SAA_dialogitemGetStateValue(context, NG_ITEM_MOTION_RADIAL, &radial);
   if (result != SI_SUCCESS) return result;
   result = SAA_dialogitemGetStateValue(context, NG_ITEM_MOTION_FLUX, &flux);
   if (result != SI_SUCCESS) return result;
   result = SAA_dialogitemGetStateValue(context, NG_ITEM_MOTION_STATIC, &staticMotion);
   if (result != SI_SUCCESS) return result;

   parameters->nodeCount = ngClampInt(parameters->nodeCount, 4, 24);
   parameters->neighbours = ngClampInt(parameters->neighbours, 1, parameters->nodeCount - 1);
   parameters->propagationSteps = ngClampInt(parameters->propagationSteps, 1, 8);
   parameters->epochs = ngClampInt(parameters->epochs, 4, NG_MAX_EPOCHS);
   parameters->learningRate = ngClampFloat(parameters->learningRate, 0.01f, 1.0f);
   parameters->layoutRadius = ngClampFloat(parameters->layoutRadius, 1.0f, 20.0f);
   parameters->nodeRadius = ngClampFloat(parameters->nodeRadius, 0.05f, 2.0f);
   parameters->edgeRadius = ngClampFloat(parameters->edgeRadius, 0.01f, 0.50f);
   parameters->weightScale = ngClampFloat(parameters->weightScale, 0.10f, 5.0f);
   parameters->randomSeed = ngClampInt(parameters->randomSeed, 0, 999999);
   parameters->framesPerEpoch = ngClampInt(parameters->framesPerEpoch, 1, 10);
   parameters->motionAmount = ngClampFloat(parameters->motionAmount, 0.0f, 1.0f);
   parameters->nodeScaleGain = ngClampFloat(parameters->nodeScaleGain, 0.0f, 3.0f);
   parameters->edgeThicknessGain = ngClampFloat(parameters->edgeThicknessGain, 0.10f, 4.0f);
   parameters->gradientSegments = ngClampInt(parameters->gradientSegments, 1, NG_MAX_GRADIENT_SEGMENTS);
   parameters->pulseCount = ngClampInt(parameters->pulseCount, 0, NG_MAX_PULSES);
   parameters->gapFrames = ngClampInt(parameters->gapFrames, 0, 60);
   parameters->diffusionSteps = ngClampInt(parameters->diffusionSteps, 1, NG_MAX_DIFFUSION_STEPS);
   parameters->framesPerDiffusion = ngClampInt(parameters->framesPerDiffusion, 1, 10);

   if (!spherical && !ring)
   {
      spherical = TRUE;
      SAA_dialogitemSetStateValue(context, NG_ITEM_LAYOUT_SPHERE, TRUE);
   }
   parameters->sphericalLayout = spherical;

   if (!radial && !flux && !staticMotion)
   {
      flux = TRUE;
      SAA_dialogitemSetStateValue(context, NG_ITEM_MOTION_FLUX, TRUE);
   }
   if (staticMotion) parameters->motionMode = NG_MOTION_STATIC;
   else if (radial) parameters->motionMode = NG_MOTION_RADIAL;
   else parameters->motionMode = NG_MOTION_FLUX;

   SAA_dialogitemSetIntValue(context, NG_ITEM_NODE_COUNT, parameters->nodeCount);
   SAA_dialogitemSetIntValue(context, NG_ITEM_NEIGHBOURS, parameters->neighbours);
   SAA_dialogitemSetIntValue(context, NG_ITEM_PROP_STEPS, parameters->propagationSteps);
   SAA_dialogitemSetIntValue(context, NG_ITEM_EPOCHS, parameters->epochs);
   SAA_dialogitemSetFloatValue(context, NG_ITEM_LEARNING_RATE, parameters->learningRate);
   SAA_dialogitemSetFloatValue(context, NG_ITEM_LAYOUT_RADIUS, parameters->layoutRadius);
   SAA_dialogitemSetFloatValue(context, NG_ITEM_NODE_RADIUS, parameters->nodeRadius);
   SAA_dialogitemSetFloatValue(context, NG_ITEM_EDGE_RADIUS, parameters->edgeRadius);
   SAA_dialogitemSetFloatValue(context, NG_ITEM_WEIGHT_SCALE, parameters->weightScale);
   SAA_dialogitemSetIntValue(context, NG_ITEM_RANDOM_SEED, parameters->randomSeed);
   SAA_dialogitemSetIntValue(context, NG_ITEM_FRAMES_PER_EPOCH, parameters->framesPerEpoch);
   SAA_dialogitemSetFloatValue(context, NG_ITEM_MOTION_AMOUNT, parameters->motionAmount);
   SAA_dialogitemSetFloatValue(context, NG_ITEM_NODE_SCALE_GAIN, parameters->nodeScaleGain);
   SAA_dialogitemSetFloatValue(context, NG_ITEM_EDGE_THICKNESS_GAIN, parameters->edgeThicknessGain);
   SAA_dialogitemSetIntValue(context, NG_ITEM_GRADIENT_SEGMENTS, parameters->gradientSegments);
   SAA_dialogitemSetIntValue(context, NG_ITEM_PULSE_COUNT, parameters->pulseCount);
   SAA_dialogitemSetIntValue(context, NG_ITEM_GAP_FRAMES, parameters->gapFrames);
   SAA_dialogitemSetIntValue(context, NG_ITEM_DIFFUSION_STEPS, parameters->diffusionSteps);
   SAA_dialogitemSetIntValue(context, NG_ITEM_FRAMES_PER_DIFFUSION, parameters->framesPerDiffusion);

   return SI_SUCCESS;
}

/* ------------------------------------------------------------------------- */
/* Scene generation                                                          */
/* ------------------------------------------------------------------------- */

static SI_Error ngGenerateAnimation(
   const SAA_CustomContext context,
   const NG_Parameters *parameters)
{
   NG_TrainingResult training;
   NG_AnimationData animation;
   NG_VisualMetrics metrics;
   NG_NodeTemplate nodeTemplate;
   NG_MaterialList materials;
   SAA_Elem root, nodesRoot, edgesRoot, pulsesRoot;
   SI_Error result;
   int startFrame, currentEndFrame, finalFrame;
   int edgeCount, actualPulseCount, pulseEdgeIndices[NG_MAX_PULSES];
   int materialCapacity, i;
   unsigned long timePart;
   char tag[32], name[80], status[NG_STATUS_BUFFER];
   SAA_Boolean rootCreated, endFrameChanged;

   memset(&training, 0, sizeof(training));
   memset(&animation, 0, sizeof(animation));
   memset(&metrics, 0, sizeof(metrics));
   memset(&nodeTemplate, 0, sizeof(nodeTemplate));
   memset(&materials, 0, sizeof(materials));
   memset(&root, 0, sizeof(root));
   memset(&nodesRoot, 0, sizeof(nodesRoot));
   memset(&edgesRoot, 0, sizeof(edgesRoot));
   memset(&pulsesRoot, 0, sizeof(pulsesRoot));
   rootCreated = FALSE;
   endFrameChanged = FALSE;

   ngTrainGraph(parameters, &training);
   if (training.epochNodeValues == NULL ||
       training.epochEdgeWeights == NULL ||
       training.epochLosses == NULL)
      return SI_ERR_ALLOC_PROBLEM;

   edgeCount = training.edgeCount;
   if (edgeCount <= 0)
   {
      ngFreeTrainingResult(&training);
      return SI_ERR_WRONG_COUNT;
   }

   result = SAA_sceneGetPlayCtrlCurrentFrame(&g_scene, &startFrame);
   if (result != SI_SUCCESS) goto fail;

   result = ngBuildAnimationData(parameters, &training, startFrame, &animation);
   if (result != SI_SUCCESS) goto fail;

   ngComputeVisualMetrics(parameters, &training, &animation, &metrics);
   finalFrame = animation.frames[animation.keyCount - 1];

   result = SAA_sceneGetPlayCtrlEndFrame(&g_scene, &currentEndFrame);
   if (result != SI_SUCCESS) goto fail;
   if (finalFrame > currentEndFrame)
   {
      result = SAA_sceneSetPlayCtrlEndFrame(&g_scene, finalFrame);
      if (result != SI_SUCCESS) goto fail;
      endFrameChanged = TRUE;
   }

   actualPulseCount = ngSelectStrongestEdges(
      parameters, &training, pulseEdgeIndices);
   materialCapacity = parameters->nodeCount +
      edgeCount * parameters->gradientSegments + actualPulseCount;
   result = ngMaterialListInit(&materials, materialCapacity);
   if (result != SI_SUCCESS) goto fail;

   ++g_graphSerial;
   timePart = ((unsigned long)time(NULL)) % 100000UL;
   sprintf(tag, "F%05lu_%02lu", timePart, g_graphSerial % 100UL);

   result = SAA_nullCreate(&g_scene, &root);
   if (result != SI_SUCCESS) goto fail;
   rootCreated = TRUE;
   sprintf(name, "NeuralGraphFlow_%s", tag);
   result = SAA_elementSetName(&g_scene, &root, name);
   if (result != SI_SUCCESS) goto fail;

   result = SAA_nullCreate(&g_scene, &nodesRoot);
   if (result != SI_SUCCESS) goto fail;
   result = SAA_modelSetParent(&g_scene, &nodesRoot, &root);
   if (result != SI_SUCCESS)
   {
      SAA_elementDestroy(&g_scene, &nodesRoot);
      goto fail;
   }
   sprintf(name, "NGF_Nodes_%s", tag);
   result = SAA_elementSetName(&g_scene, &nodesRoot, name);
   if (result != SI_SUCCESS) goto fail;

   result = SAA_nullCreate(&g_scene, &edgesRoot);
   if (result != SI_SUCCESS) goto fail;
   result = SAA_modelSetParent(&g_scene, &edgesRoot, &root);
   if (result != SI_SUCCESS)
   {
      SAA_elementDestroy(&g_scene, &edgesRoot);
      goto fail;
   }
   sprintf(name, "NGF_EdgeSegments_%s", tag);
   result = SAA_elementSetName(&g_scene, &edgesRoot, name);
   if (result != SI_SUCCESS) goto fail;

   if (actualPulseCount > 0)
   {
      result = SAA_nullCreate(&g_scene, &pulsesRoot);
      if (result != SI_SUCCESS) goto fail;
      result = SAA_modelSetParent(&g_scene, &pulsesRoot, &root);
      if (result != SI_SUCCESS)
      {
         SAA_elementDestroy(&g_scene, &pulsesRoot);
         goto fail;
      }
      sprintf(name, "NGF_FlowPulses_%s", tag);
      result = SAA_elementSetName(&g_scene, &pulsesRoot, name);
      if (result != SI_SUCCESS) goto fail;
   }

   /* Cache a unit sphere topology for nodes and pulse beads. */
   {
      SAA_Elem tempSphere;
      memset(&tempSphere, 0, sizeof(tempSphere));
      result = SAA_meshCreateSphere(&g_scene, 1.0f, 8, 8, &tempSphere);
      if (result != SI_SUCCESS) goto fail;
      result = ngGetTemplateFromSphere(&g_scene, &tempSphere, &nodeTemplate);
      SAA_elementDestroy(&g_scene, &tempSphere);
      if (result != SI_SUCCESS) goto fail;
   }

   /* Nodes: shape keys encode activation scale/motion; diffuse F-curves encode
      evolving signed activation colors. */
   for (i = 0; i < parameters->nodeCount; ++i)
   {
      SAA_Elem object, material;
      SAA_DVector *vertices;
      NG_Color *colors;
      NG_Vector3 position;
      double radius;
      int key;

      memset(&object, 0, sizeof(object));
      memset(&material, 0, sizeof(material));
      vertices = (SAA_DVector *)malloc(sizeof(SAA_DVector) * nodeTemplate.count);
      colors = (NG_Color *)malloc(sizeof(NG_Color) * animation.keyCount);
      if (vertices == NULL || colors == NULL)
      {
         free(vertices); free(colors);
         result = SI_ERR_ALLOC_PROBLEM;
         goto fail;
      }

      ngBuildAnimatedNodePose(
         parameters, &training, animation.nodeValues, animation.edgeWeights,
         i, &position, &radius);
      ngBuildNodeVertices(&nodeTemplate, position, radius, vertices);

      result = SAA_meshCreateSphere(&g_scene, 1.0f, 8, 8, &object);
      if (result != SI_SUCCESS)
      {
         free(vertices); free(colors); goto fail;
      }
      result = SAA_modelSetVertices(
         &g_scene, &object, SAA_GEOM_ORIGINAL, 0,
         nodeTemplate.count, vertices);
      free(vertices);
      if (result != SI_SUCCESS)
      {
         free(colors); SAA_elementDestroy(&g_scene, &object); goto fail;
      }
      result = SAA_modelSetParent(&g_scene, &object, &nodesRoot);
      if (result != SI_SUCCESS)
      {
         free(colors); SAA_elementDestroy(&g_scene, &object); goto fail;
      }
      sprintf(name, "FN%02d_%s", i, tag);
      result = SAA_elementSetName(&g_scene, &object, name);
      if (result != SI_SUCCESS) { free(colors); goto fail; }
      result = ngPrepareMeshNormals(&g_scene, &object);
      if (result != SI_SUCCESS) { free(colors); goto fail; }

      for (key = 0; key < animation.keyCount; ++key)
         ngActivationColor(
            animation.nodeValues[key * NG_MAX_NODES + i],
            metrics.maximumNodeValue, &colors[key]);

      sprintf(name, "FMN%02d_%s", i, tag);
      result = ngCreateAnimatedMaterial(
         &g_scene, name, &animation, colors, &material);
      free(colors);
      if (result != SI_SUCCESS) goto fail;
      result = ngMaterialListAdd(&materials, &material);
      if (result != SI_SUCCESS)
      {
         SAA_elementDestroy(&g_scene, &material); goto fail;
      }
      result = ngAssignGlobalMaterial(&g_scene, &object, &material);
      if (result != SI_SUCCESS) goto fail;
      result = ngBakeNodeShapes(
         parameters, &training, &animation, &nodeTemplate, &object, i);
      if (result != SI_SUCCESS) goto fail;

      sprintf(status, "NeuralGraph v0.3: baked node %d/%d",
         i + 1, parameters->nodeCount);
      SAA_statusBarSet(status, SAA_MESSAGE_CODE);
   }

   /* Every logical edge is split into a configurable number of open prisms.
      Their material colors interpolate between endpoint activations. */
   for (i = 0; i < edgeCount; ++i)
   {
      int segment;
      for (segment = 0; segment < parameters->gradientSegments; ++segment)
      {
         SAA_Elem object, material;
         NG_Color *colors;
         NG_Vector3 positionA, positionB, start, end, segmentStart, segmentEnd;
         double radiusA, radiusB, edgeRadius, t;
         int a, b, key;

         memset(&object, 0, sizeof(object));
         memset(&material, 0, sizeof(material));
         a = training.edges[i].a;
         b = training.edges[i].b;

         ngBuildAnimatedNodePose(
            parameters, &training, animation.nodeValues, animation.edgeWeights,
            a, &positionA, &radiusA);
         ngBuildAnimatedNodePose(
            parameters, &training, animation.nodeValues, animation.edgeWeights,
            b, &positionB, &radiusB);
         ngComputeShortenedEdge(
            positionA, positionB, radiusA, radiusB, &start, &end);
         ngComputeSegmentEndpoints(
            start, end, segment, parameters->gradientSegments,
            &segmentStart, &segmentEnd);
         edgeRadius = ngComputeEdgeRadius(
            parameters, animation.edgeWeights[i], metrics.maximumEdgeWeight);

         result = ngCreateEdgeMesh(
            &g_scene, segmentStart, segmentEnd, edgeRadius, &object);
         if (result != SI_SUCCESS) goto fail;
         result = SAA_modelSetParent(&g_scene, &object, &edgesRoot);
         if (result != SI_SUCCESS)
         {
            SAA_elementDestroy(&g_scene, &object); goto fail;
         }
         sprintf(name, "FE%02d_%02d_S%d_%s", a, b, segment, tag);
         result = SAA_elementSetName(&g_scene, &object, name);
         if (result != SI_SUCCESS) goto fail;

         colors = (NG_Color *)malloc(sizeof(NG_Color) * animation.keyCount);
         if (colors == NULL)
         {
            result = SI_ERR_ALLOC_PROBLEM; goto fail;
         }
         t = ((double)segment + 0.5) / (double)parameters->gradientSegments;
         for (key = 0; key < animation.keyCount; ++key)
         {
            ngEdgeGradientColor(
               animation.nodeValues[key * NG_MAX_NODES + a],
               animation.nodeValues[key * NG_MAX_NODES + b],
               animation.edgeWeights[key * NG_MAX_EDGES + i],
               t, &metrics, &colors[key]);
         }

         sprintf(name, "FME%02d_%02d_S%d_%s", a, b, segment, tag);
         result = ngCreateAnimatedMaterial(
            &g_scene, name, &animation, colors, &material);
         free(colors);
         if (result != SI_SUCCESS) goto fail;
         result = ngMaterialListAdd(&materials, &material);
         if (result != SI_SUCCESS)
         {
            SAA_elementDestroy(&g_scene, &material); goto fail;
         }
         result = ngAssignGlobalMaterial(&g_scene, &object, &material);
         if (result != SI_SUCCESS) goto fail;
         result = ngBakeEdgeSegmentShapes(
            parameters, &training, &animation, &metrics,
            &object, i, segment);
         if (result != SI_SUCCESS) goto fail;
      }

      sprintf(status, "NeuralGraph v0.3: baked gradient edge %d/%d",
         i + 1, edgeCount);
      SAA_statusBarSet(status, SAA_MESSAGE_CODE);
   }

   /* Pulse beads travel over the strongest final learned connections during
      the PDE playback phase. */
   for (i = 0; i < actualPulseCount; ++i)
   {
      SAA_Elem object, material;
      SAA_DVector *vertices;
      NG_Color *colors;
      NG_Vector3 position;
      double radius;
      float flow;
      int key, edgeIndex;

      memset(&object, 0, sizeof(object));
      memset(&material, 0, sizeof(material));
      edgeIndex = pulseEdgeIndices[i];
      vertices = (SAA_DVector *)malloc(sizeof(SAA_DVector) * nodeTemplate.count);
      colors = (NG_Color *)malloc(sizeof(NG_Color) * animation.keyCount);
      if (vertices == NULL || colors == NULL)
      {
         free(vertices); free(colors);
         result = SI_ERR_ALLOC_PROBLEM; goto fail;
      }

      ngBuildPulsePose(
         parameters, &training, &animation, &metrics,
         0, edgeIndex, i, &position, &radius, &flow);
      ngBuildNodeVertices(&nodeTemplate, position, radius, vertices);

      result = SAA_meshCreateSphere(&g_scene, 1.0f, 8, 8, &object);
      if (result != SI_SUCCESS)
      {
         free(vertices); free(colors); goto fail;
      }
      result = SAA_modelSetVertices(
         &g_scene, &object, SAA_GEOM_ORIGINAL, 0,
         nodeTemplate.count, vertices);
      free(vertices);
      if (result != SI_SUCCESS)
      {
         free(colors); SAA_elementDestroy(&g_scene, &object); goto fail;
      }
      result = SAA_modelSetParent(&g_scene, &object, &pulsesRoot);
      if (result != SI_SUCCESS)
      {
         free(colors); SAA_elementDestroy(&g_scene, &object); goto fail;
      }
      sprintf(name, "FP%02d_%s", i, tag);
      result = SAA_elementSetName(&g_scene, &object, name);
      if (result != SI_SUCCESS) { free(colors); goto fail; }
      result = ngPrepareMeshNormals(&g_scene, &object);
      if (result != SI_SUCCESS) { free(colors); goto fail; }

      for (key = 0; key < animation.keyCount; ++key)
      {
         ngBuildPulsePose(
            parameters, &training, &animation, &metrics,
            key, edgeIndex, i, &position, &radius, &flow);
         ngPulseColor(animation.phases[key], flow,
            metrics.maximumFlow, &colors[key]);
      }

      sprintf(name, "FMP%02d_%s", i, tag);
      result = ngCreateAnimatedMaterial(
         &g_scene, name, &animation, colors, &material);
      free(colors);
      if (result != SI_SUCCESS) goto fail;
      result = ngMaterialListAdd(&materials, &material);
      if (result != SI_SUCCESS)
      {
         SAA_elementDestroy(&g_scene, &material); goto fail;
      }
      result = ngAssignGlobalMaterial(&g_scene, &object, &material);
      if (result != SI_SUCCESS) goto fail;
      result = ngBakePulseShapes(
         parameters, &training, &animation, &metrics,
         &nodeTemplate, &object, edgeIndex, i);
      if (result != SI_SUCCESS) goto fail;
   }

   result = SAA_sceneSetPlayCtrlCurrentFrame(&g_scene, startFrame);
   if (result != SI_SUCCESS) goto fail;
   SAA_selectlistClear(&g_scene);
   result = SAA_selectlistSetModelBranch(&g_scene, TRUE, 1, &root);
   if (result != SI_SUCCESS) goto fail;
   result = SAA_sceneRefresh(context);
   if (result != SI_SUCCESS) goto fail;

   sprintf(g_statusComplete,
      "NeuralGraph v0.3: loss %.5f -> %.5f; frames %d-%d.",
      training.epochLosses[0],
      training.epochLosses[parameters->epochs],
      animation.frames[0], animation.frames[animation.keyCount - 1]);

   ngFreeNodeTemplate(&nodeTemplate);
   ngMaterialListRelease(&g_scene, &materials, FALSE);
   ngFreeAnimationData(&animation);
   ngFreeTrainingResult(&training);
   return SI_SUCCESS;

fail:
   if (endFrameChanged)
      SAA_sceneSetPlayCtrlEndFrame(&g_scene, currentEndFrame);
   ngFreeNodeTemplate(&nodeTemplate);
   ngFreeAnimationData(&animation);
   ngFreeTrainingResult(&training);
   ngCleanupPartialGraph(&g_scene, rootCreated, &root, &materials);
   SAA_sceneRefresh(context);
   return result;
}
/* ------------------------------------------------------------------------- */
/* Layout and topology                                                       */
/* ------------------------------------------------------------------------- */

static void ngBuildLayout(
   const NG_Parameters *parameters,
   NG_Random *random,
   NG_Vector3 *positions)
{
   int i;
   double phase;

   phase = ngRandomUnit(random) * 2.0 * NG_PI;

   for (i = 0; i < parameters->nodeCount; ++i)
   {
      if (parameters->sphericalLayout)
      {
         double y = 1.0 -
            2.0 * (double)i /
            (double)(parameters->nodeCount - 1);
         double radial = sqrt(ngMax(0.0, 1.0 - y * y));
         double theta = NG_GOLDEN_ANGLE * (double)i + phase;

         positions[i].x = parameters->layoutRadius * radial * cos(theta);
         positions[i].y = parameters->layoutRadius * y;
         positions[i].z = parameters->layoutRadius * radial * sin(theta);
      }
      else
      {
         double theta =
            2.0 * NG_PI * (double)i / (double)parameters->nodeCount + phase;

         positions[i].x = parameters->layoutRadius * cos(theta);
         positions[i].y = parameters->layoutRadius * 0.18 * sin(2.0 * theta + 0.65);
         positions[i].z = parameters->layoutRadius * sin(theta);
      }
   }
}

static int ngBuildEdges(
   const NG_Parameters *parameters,
   NG_Random *random,
   const NG_Vector3 *positions,
   NG_Edge *edges)
{
   int connected[NG_MAX_NODES][NG_MAX_NODES];
   int edgeCount;
   int i;
   int pass;
   double sigma;

   memset(connected, 0, sizeof(connected));
   edgeCount = 0;
   sigma = ngMax(parameters->layoutRadius * 0.90, 0.1);

   for (i = 0; i < parameters->nodeCount; ++i)
   {
      int chosen[NG_MAX_NODES];
      memset(chosen, 0, sizeof(chosen));

      for (pass = 0; pass < parameters->neighbours; ++pass)
      {
         int j;
         int best;
         double bestDistance;

         best = -1;
         bestDistance = 1.0e100;

         for (j = 0; j < parameters->nodeCount; ++j)
         {
            double distanceSquared;

            if (j == i || chosen[j])
               continue;

            distanceSquared = ngDistanceSquared(positions[i], positions[j]);
            if (distanceSquared < bestDistance)
            {
               bestDistance = distanceSquared;
               best = j;
            }
         }

         if (best < 0)
            break;

         chosen[best] = 1;

         if (connected[i][best])
            continue;

         connected[i][best] = 1;
         connected[best][i] = 1;

         if (edgeCount < NG_MAX_EDGES)
         {
            double proximity = exp(-bestDistance / (2.0 * sigma * sigma));
            double magnitude = 0.18 + 0.72 * proximity;
            double jitter = 0.75 + 0.50 * ngRandomUnit(random);
            double weight = parameters->weightScale * magnitude * jitter;

            if (parameters->signedWeights && ngRandomUnit(random) < 0.45f)
               weight = -weight;

            edges[edgeCount].a = i;
            edges[edgeCount].b = best;
            edges[edgeCount].weight = (float)weight;
            edges[edgeCount].distanceSquared = (float)bestDistance;
            ++edgeCount;
         }
      }
   }

   return edgeCount;
}

static void ngBuildSourceAndTargetFields(
   const NG_Parameters *parameters,
   const NG_Vector3 *positions,
   float *sourceField,
   float *targetField)
{
   int i;
   int srcPos;
   int srcNeg;
   double maxY;
   double minY;
   double sigmaTarget;

   srcPos = 0;
   srcNeg = parameters->nodeCount > 1 ? parameters->nodeCount - 1 : 0;
   maxY = positions[0].y;
   minY = positions[0].y;

   for (i = 1; i < parameters->nodeCount; ++i)
   {
      if (positions[i].y > maxY)
      {
         maxY = positions[i].y;
         srcPos = i;
      }
      if (positions[i].y < minY)
      {
         minY = positions[i].y;
         srcNeg = i;
      }
   }

   sigmaTarget = ngMax(parameters->layoutRadius * 0.85, 0.50);

   for (i = 0; i < parameters->nodeCount; ++i)
   {
      double dPos;
      double dNeg;
      double tPos;
      double tNeg;

      sourceField[i] = 0.0f;
      if (i == srcPos) sourceField[i] += 1.0f;
      if (i == srcNeg) sourceField[i] -= 1.0f;

      dPos = ngDistanceSquared(positions[i], positions[srcPos]);
      dNeg = ngDistanceSquared(positions[i], positions[srcNeg]);
      tPos = exp(-dPos / (2.0 * sigmaTarget * sigmaTarget));
      tNeg = exp(-dNeg / (2.0 * sigmaTarget * sigmaTarget));

      targetField[i] = (float)(tPos - 0.85 * tNeg);
   }
}

/* ------------------------------------------------------------------------- */
/* Training                                                                  */
/* ------------------------------------------------------------------------- */

static void ngForwardPropagate(
   const NG_Parameters *parameters,
   const NG_Edge *edges,
   int edgeCount,
   const float *sourceField,
   float *stateOut)
{
   float current[NG_MAX_NODES];
   float next[NG_MAX_NODES];
   int degree[NG_MAX_NODES];
   int step;
   int i;

   for (i = 0; i < parameters->nodeCount; ++i)
      current[i] = sourceField[i];

   for (step = 0; step < parameters->propagationSteps; ++step)
   {
      float sum[NG_MAX_NODES];
      memset(sum, 0, sizeof(sum));
      memset(degree, 0, sizeof(degree));

      for (i = 0; i < edgeCount; ++i)
      {
         int a = edges[i].a;
         int b = edges[i].b;
         float w = edges[i].weight;

         sum[a] += w * current[b];
         sum[b] += w * current[a];
         ++degree[a];
         ++degree[b];
      }

      for (i = 0; i < parameters->nodeCount; ++i)
      {
         double aggregate;
         aggregate = degree[i] > 0 ? sum[i] / (double)degree[i] : 0.0;
         next[i] = (float)tanh(
            0.60 * sourceField[i] +
            0.28 * current[i] +
            0.72 * aggregate);
      }

      for (i = 0; i < parameters->nodeCount; ++i)
         current[i] = next[i];
   }

   for (i = 0; i < parameters->nodeCount; ++i)
      stateOut[i] = current[i];
}

static float ngComputeLoss(
   int count,
   const float *predicted,
   const float *target)
{
   int i;
   double loss;

   loss = 0.0;
   for (i = 0; i < count; ++i)
   {
      double d = target[i] - predicted[i];
      loss += d * d;
   }

   if (count <= 0) count = 1;
   return (float)(loss / (double)count);
}

static void ngOptimizeWeights(
   const NG_Parameters *parameters,
   NG_TrainingResult *training,
   int epoch)
{
   float predicted[NG_MAX_NODES];
   float trialPredicted[NG_MAX_NODES];
   float bestLoss;
   float stepScale;
   float clampMagnitude;
   int i;

   /*
    * Deterministic derivative-free coordinate descent.
    *
    * For every edge, test a positive and negative weight perturbation while
    * keeping all previously accepted changes from this epoch. Accept only a
    * candidate that lowers the global mean-squared error. This makes the
    * training history monotonic apart from floating-point roundoff.
    */
   ngForwardPropagate(
      parameters,
      training->edges,
      training->edgeCount,
      training->sourceField,
      predicted);

   bestLoss = ngComputeLoss(
      parameters->nodeCount,
      predicted,
      training->targetField);

   if (parameters->epochs > 1)
   {
      double progress = (double)epoch / (double)(parameters->epochs - 1);
      stepScale = parameters->learningRate * parameters->weightScale *
         (float)(0.75 - 0.55 * progress);
   }
   else
   {
      stepScale = parameters->learningRate * parameters->weightScale * 0.75f;
   }

   if (stepScale < 0.0001f)
      stepScale = 0.0001f;

   clampMagnitude = parameters->weightScale * 2.75f;

   for (i = 0; i < training->edgeCount; ++i)
   {
      float originalWeight = training->edges[i].weight;
      float bestWeight = originalWeight;
      int direction;

      for (direction = -1; direction <= 1; direction += 2)
      {
         float candidate = originalWeight + direction * stepScale;
         float candidateLoss;

         if (parameters->signedWeights)
         {
            candidate = ngClampFloat(
               candidate, -clampMagnitude, clampMagnitude);
         }
         else
         {
            candidate = ngClampFloat(
               candidate, 0.01f, clampMagnitude);
         }

         training->edges[i].weight = candidate;

         ngForwardPropagate(
            parameters,
            training->edges,
            training->edgeCount,
            training->sourceField,
            trialPredicted);

         candidateLoss = ngComputeLoss(
            parameters->nodeCount,
            trialPredicted,
            training->targetField);

         if (candidateLoss + 1.0e-9f < bestLoss)
         {
            bestLoss = candidateLoss;
            bestWeight = candidate;
         }
      }

      training->edges[i].weight = bestWeight;
   }
}

static void ngTrainGraph(
   const NG_Parameters *parameters,
   NG_TrainingResult *training)
{
   NG_Random random;
   int epoch;

   memset(training, 0, sizeof(*training));

   training->epochNodeValues =
      (float *)calloc((parameters->epochs + 1) * NG_MAX_NODES, sizeof(float));
   training->epochEdgeWeights =
      (float *)calloc((parameters->epochs + 1) * NG_MAX_EDGES, sizeof(float));
   training->epochLosses =
      (float *)calloc(parameters->epochs + 1, sizeof(float));

   if (training->epochNodeValues == NULL ||
       training->epochEdgeWeights == NULL ||
       training->epochLosses == NULL)
   {
      ngFreeTrainingResult(training);
      return;
   }

   random.state = ((unsigned long)parameters->randomSeed) ^ 0xA341316CUL;
   if (random.state == 0)
      random.state = 1;

   ngBuildLayout(parameters, &random, training->positions);
   training->edgeCount = ngBuildEdges(parameters, &random, training->positions, training->edges);
   ngBuildSourceAndTargetFields(
      parameters,
      training->positions,
      training->sourceField,
      training->targetField);

   if (training->edgeCount <= 0)
      return;

   for (epoch = 0; epoch <= parameters->epochs; ++epoch)
   {
      float predicted[NG_MAX_NODES];
      int i;

      ngForwardPropagate(
         parameters,
         training->edges,
         training->edgeCount,
         training->sourceField,
         predicted);

      for (i = 0; i < parameters->nodeCount; ++i)
         training->epochNodeValues[epoch * NG_MAX_NODES + i] = predicted[i];
      for (i = 0; i < training->edgeCount; ++i)
         training->epochEdgeWeights[epoch * NG_MAX_EDGES + i] = training->edges[i].weight;

      training->epochLosses[epoch] = ngComputeLoss(
         parameters->nodeCount, predicted, training->targetField);

      if (epoch < parameters->epochs)
         ngOptimizeWeights(parameters, training, epoch);
   }
}


/* ------------------------------------------------------------------------- */
/* Animation timeline and PDE playback                                       */
/* ------------------------------------------------------------------------- */

static SI_Error ngBuildAnimationData(
   const NG_Parameters *parameters,
   const NG_TrainingResult *training,
   int startFrame,
   NG_AnimationData *animation)
{
   int capacity, key, epoch, step, finalTrainingFrame, playbackStartFrame;
   float *diffusionStates;
   SI_Error result;

   if (animation == NULL) return SI_ERR_BAD_ARGUMENT;
   memset(animation, 0, sizeof(*animation));

   capacity = parameters->epochs + 1 +
      (parameters->gapFrames > 0 ? 1 : 0) +
      parameters->diffusionSteps + 1;

   animation->frames = (int *)calloc(capacity, sizeof(int));
   animation->times = (float *)calloc(capacity, sizeof(float));
   animation->phases = (int *)calloc(capacity, sizeof(int));
   animation->phaseSteps = (int *)calloc(capacity, sizeof(int));
   animation->nodeValues = (float *)calloc(
      capacity * NG_MAX_NODES, sizeof(float));
   animation->edgeWeights = (float *)calloc(
      capacity * NG_MAX_EDGES, sizeof(float));

   if (animation->frames == NULL || animation->times == NULL ||
       animation->phases == NULL || animation->phaseSteps == NULL ||
       animation->nodeValues == NULL || animation->edgeWeights == NULL)
   {
      ngFreeAnimationData(animation);
      return SI_ERR_ALLOC_PROBLEM;
   }

   key = 0;
   for (epoch = 0; epoch <= parameters->epochs; ++epoch)
   {
      int i;
      animation->frames[key] =
         startFrame + epoch * parameters->framesPerEpoch;
      animation->phases[key] = NG_PHASE_TRAINING;
      animation->phaseSteps[key] = epoch;
      for (i = 0; i < parameters->nodeCount; ++i)
         animation->nodeValues[key * NG_MAX_NODES + i] =
            training->epochNodeValues[epoch * NG_MAX_NODES + i];
      for (i = 0; i < training->edgeCount; ++i)
         animation->edgeWeights[key * NG_MAX_EDGES + i] =
            training->epochEdgeWeights[epoch * NG_MAX_EDGES + i];
      ++key;
   }

   finalTrainingFrame = startFrame + parameters->epochs * parameters->framesPerEpoch;
   if (parameters->gapFrames > 0)
   {
      int i;
      animation->frames[key] = finalTrainingFrame + parameters->gapFrames;
      animation->phases[key] = NG_PHASE_HOLD;
      animation->phaseSteps[key] = 0;
      for (i = 0; i < parameters->nodeCount; ++i)
         animation->nodeValues[key * NG_MAX_NODES + i] =
            training->epochNodeValues[parameters->epochs * NG_MAX_NODES + i];
      for (i = 0; i < training->edgeCount; ++i)
         animation->edgeWeights[key * NG_MAX_EDGES + i] =
            training->epochEdgeWeights[parameters->epochs * NG_MAX_EDGES + i];
      ++key;
   }

   diffusionStates = (float *)calloc(
      (parameters->diffusionSteps + 1) * NG_MAX_NODES, sizeof(float));
   if (diffusionStates == NULL)
   {
      ngFreeAnimationData(animation);
      return SI_ERR_ALLOC_PROBLEM;
   }
   ngSimulateDiffusion(parameters, training, diffusionStates);

   playbackStartFrame = finalTrainingFrame + parameters->gapFrames + 1;
   for (step = 0; step <= parameters->diffusionSteps; ++step)
   {
      int i;
      animation->frames[key] =
         playbackStartFrame + step * parameters->framesPerDiffusion;
      animation->phases[key] = NG_PHASE_DIFFUSION;
      animation->phaseSteps[key] = step;
      for (i = 0; i < parameters->nodeCount; ++i)
         animation->nodeValues[key * NG_MAX_NODES + i] =
            diffusionStates[step * NG_MAX_NODES + i];
      for (i = 0; i < training->edgeCount; ++i)
         animation->edgeWeights[key * NG_MAX_EDGES + i] =
            training->epochEdgeWeights[parameters->epochs * NG_MAX_EDGES + i];
      ++key;
   }
   free(diffusionStates);
   animation->keyCount = key;

   for (key = 0; key < animation->keyCount; ++key)
   {
      result = SAA_frame2Seconds(
         &g_scene, animation->frames[key], &animation->times[key]);
      if (result != SI_SUCCESS)
      {
         ngFreeAnimationData(animation);
         return result;
      }
   }
   return SI_SUCCESS;
}

static void ngSimulateDiffusion(
   const NG_Parameters *parameters,
   const NG_TrainingResult *training,
   float *states)
{
   float current[NG_MAX_NODES], next[NG_MAX_NODES];
   float maximumWeight;
   int step, i;

   maximumWeight = 0.0001f;
   for (i = 0; i < training->edgeCount; ++i)
   {
      float w = (float)ngAbs(training->epochEdgeWeights[
         parameters->epochs * NG_MAX_EDGES + i]);
      if (w > maximumWeight) maximumWeight = w;
   }

   for (i = 0; i < parameters->nodeCount; ++i)
   {
      current[i] = training->sourceField[i];
      states[i] = current[i];
   }

   for (step = 1; step <= parameters->diffusionSteps; ++step)
   {
      float delta[NG_MAX_NODES], conductanceSum[NG_MAX_NODES];
      memset(delta, 0, sizeof(delta));
      memset(conductanceSum, 0, sizeof(conductanceSum));

      for (i = 0; i < training->edgeCount; ++i)
      {
         int a = training->edges[i].a;
         int b = training->edges[i].b;
         float w = training->epochEdgeWeights[
            parameters->epochs * NG_MAX_EDGES + i];
         float conductance =
            0.05f + 0.95f * ((float)ngAbs(w) / maximumWeight);
         delta[a] += conductance * (current[b] - current[a]);
         delta[b] += conductance * (current[a] - current[b]);
         conductanceSum[a] += conductance;
         conductanceSum[b] += conductance;
      }

      for (i = 0; i < parameters->nodeCount; ++i)
      {
         double correction = 0.0;
         if (conductanceSum[i] > 0.0001f)
            correction = delta[i] / conductanceSum[i];
         next[i] = ngClampFloat(
            current[i] + 0.58f * (float)correction, -1.15f, 1.15f);
         if (ngAbs(training->sourceField[i]) > 0.5f)
            next[i] = training->sourceField[i];
      }

      for (i = 0; i < parameters->nodeCount; ++i)
      {
         current[i] = next[i];
         states[step * NG_MAX_NODES + i] = current[i];
      }
   }
}

static void ngFreeAnimationData(NG_AnimationData *animation)
{
   if (animation == NULL) return;
   free(animation->frames);
   free(animation->times);
   free(animation->phases);
   free(animation->phaseSteps);
   free(animation->nodeValues);
   free(animation->edgeWeights);
   memset(animation, 0, sizeof(*animation));
}

static void ngComputeVisualMetrics(
   const NG_Parameters *parameters,
   const NG_TrainingResult *training,
   const NG_AnimationData *animation,
   NG_VisualMetrics *metrics)
{
   int key, i;

   metrics->maximumNodeValue = 0.0001f;
   metrics->maximumEdgeWeight = 0.0001f;
   metrics->maximumFlow = 0.0001f;

   for (key = 0; key < animation->keyCount; ++key)
   {
      for (i = 0; i < parameters->nodeCount; ++i)
      {
         float value = (float)ngAbs(
            animation->nodeValues[key * NG_MAX_NODES + i]);
         if (value > metrics->maximumNodeValue)
            metrics->maximumNodeValue = value;
      }
      for (i = 0; i < training->edgeCount; ++i)
      {
         int a = training->edges[i].a;
         int b = training->edges[i].b;
         float w = animation->edgeWeights[key * NG_MAX_EDGES + i];
         float flow = (float)ngAbs(w) *
            (animation->nodeValues[key * NG_MAX_NODES + a] -
             animation->nodeValues[key * NG_MAX_NODES + b]);
         float aw = (float)ngAbs(w);
         if (aw > metrics->maximumEdgeWeight) metrics->maximumEdgeWeight = aw;
         if (ngAbs(flow) > metrics->maximumFlow)
            metrics->maximumFlow = (float)ngAbs(flow);
      }
   }
}

/* ------------------------------------------------------------------------- */
/* Geometry motion                                                           */
/* ------------------------------------------------------------------------- */

static void ngComputeNodeFlux(
   const NG_Parameters *parameters,
   const NG_TrainingResult *training,
   const float *nodeValues,
   const float *edgeWeights,
   int nodeIndex,
   NG_Vector3 *flux)
{
   double denominator;
   int i;
   (void)parameters;

   flux->x = flux->y = flux->z = 0.0;
   denominator = 0.0;
   for (i = 0; i < training->edgeCount; ++i)
   {
      int other = -1;
      double stateDifference = 0.0;
      NG_Vector3 direction;
      double weight;

      if (training->edges[i].a == nodeIndex)
      {
         other = training->edges[i].b;
         stateDifference = nodeValues[other] - nodeValues[nodeIndex];
      }
      else if (training->edges[i].b == nodeIndex)
      {
         other = training->edges[i].a;
         stateDifference = nodeValues[other] - nodeValues[nodeIndex];
      }
      if (other < 0) continue;

      direction = ngNormalize(ngSubtract(
         training->positions[other], training->positions[nodeIndex]));
      weight = edgeWeights[i];
      *flux = ngAdd(*flux, ngScale(direction, weight * stateDifference));
      denominator += ngAbs(weight);
   }
   if (denominator > NG_EPSILON)
      *flux = ngScale(*flux, 1.0 / denominator);
}

static void ngBuildAnimatedNodePose(
   const NG_Parameters *parameters,
   const NG_TrainingResult *training,
   const float *nodeValues,
   const float *edgeWeights,
   int nodeIndex,
   NG_Vector3 *position,
   double *radius)
{
   NG_Vector3 basePosition;
   double value, magnitude;

   basePosition = training->positions[nodeIndex];
   value = nodeValues[nodeIndex];
   magnitude = ngAbs(value);
   *radius = parameters->nodeRadius *
      (0.66 + parameters->nodeScaleGain * magnitude);
   if (*radius < parameters->nodeRadius * 0.25)
      *radius = parameters->nodeRadius * 0.25;
   if (*radius > parameters->nodeRadius * 3.25)
      *radius = parameters->nodeRadius * 3.25;

   if (parameters->motionMode == NG_MOTION_RADIAL)
   {
      NG_Vector3 normal = ngNormalize(basePosition);
      double displacement = parameters->motionAmount *
         parameters->layoutRadius * tanh(value);
      *position = ngAdd(basePosition, ngScale(normal, displacement));
   }
   else if (parameters->motionMode == NG_MOTION_FLUX)
   {
      NG_Vector3 flux;
      double length, scale;
      ngComputeNodeFlux(parameters, training, nodeValues, edgeWeights,
         nodeIndex, &flux);
      length = ngLength(flux);
      scale = parameters->motionAmount * parameters->layoutRadius * 1.8 /
         (1.0 + length);
      *position = ngAdd(basePosition, ngScale(flux, scale));
   }
   else
      *position = basePosition;
}

static double ngComputeEdgeRadius(
   const NG_Parameters *parameters,
   float weight,
   float maximumWeight)
{
   double normalized;
   maximumWeight = ngClampFloat(maximumWeight, 0.0001f, 1000000.0f);
   normalized = ngAbs(weight) / maximumWeight;
   return parameters->edgeRadius *
      (0.45 + parameters->edgeThicknessGain * normalized);
}

static void ngBuildAnimatedEdgeVertices(
   NG_Vector3 start,
   NG_Vector3 end,
   double radius,
   SAA_DVector *vertices)
{
   NG_Vector3 direction, reference, u, v;
   double length;
   int i;

   direction = ngSubtract(end, start);
   length = ngLength(direction);
   if (length < 0.0001) length = 0.0001;
   direction = ngScale(direction, 1.0 / length);

   if (ngAbs(direction.y) < 0.90)
   {
      reference.x = 0.0; reference.y = 1.0; reference.z = 0.0;
   }
   else
   {
      reference.x = 1.0; reference.y = 0.0; reference.z = 0.0;
   }
   u = ngNormalize(ngCross(reference, direction));
   v = ngNormalize(ngCross(direction, u));

   for (i = 0; i < NG_EDGE_SIDES; ++i)
   {
      double angle = 2.0 * NG_PI * (double)i / (double)NG_EDGE_SIDES;
      NG_Vector3 offset = ngAdd(
         ngScale(u, radius * cos(angle)),
         ngScale(v, radius * sin(angle)));
      NG_Vector3 bottom = ngAdd(start, offset);
      NG_Vector3 top = ngAdd(end, offset);
      vertices[i].x = bottom.x; vertices[i].y = bottom.y;
      vertices[i].z = bottom.z; vertices[i].w = 1.0;
      vertices[NG_EDGE_SIDES + i].x = top.x;
      vertices[NG_EDGE_SIDES + i].y = top.y;
      vertices[NG_EDGE_SIDES + i].z = top.z;
      vertices[NG_EDGE_SIDES + i].w = 1.0;
   }
}

static void ngComputeShortenedEdge(
   NG_Vector3 positionA,
   NG_Vector3 positionB,
   double radiusA,
   double radiusB,
   NG_Vector3 *start,
   NG_Vector3 *end)
{
   NG_Vector3 fullDirection, direction;
   double fullLength, shortenA, shortenB, totalShortening;

   fullDirection = ngSubtract(positionB, positionA);
   fullLength = ngLength(fullDirection);
   if (fullLength < 0.0001)
   {
      *start = positionA;
      *end = positionA;
      end->x += 0.0001;
      return;
   }

   direction = ngScale(fullDirection, 1.0 / fullLength);
   shortenA = radiusA * 0.80;
   shortenB = radiusB * 0.80;
   totalShortening = shortenA + shortenB;
   if (totalShortening > fullLength * 0.75 && totalShortening > NG_EPSILON)
   {
      double scale = (fullLength * 0.75) / totalShortening;
      shortenA *= scale;
      shortenB *= scale;
   }
   *start = ngAdd(positionA, ngScale(direction, shortenA));
   *end = ngSubtract(positionB, ngScale(direction, shortenB));
}

static void ngComputeSegmentEndpoints(
   NG_Vector3 start,
   NG_Vector3 end,
   int segmentIndex,
   int segmentCount,
   NG_Vector3 *segmentStart,
   NG_Vector3 *segmentEnd)
{
   double t0 = (double)segmentIndex / (double)segmentCount;
   double t1 = (double)(segmentIndex + 1) / (double)segmentCount;
   *segmentStart = ngLerpVector(start, end, t0);
   *segmentEnd = ngLerpVector(start, end, t1);
}

/* ------------------------------------------------------------------------- */
/* Materials and animated gradients                                          */
/* ------------------------------------------------------------------------- */

static void ngActivationColor(float value, float maximumValue, NG_Color *color)
{
   double n, t;
   maximumValue = ngClampFloat(maximumValue, 0.0001f, 1000000.0f);
   n = ngClampFloat(value / maximumValue, -1.0f, 1.0f);
   if (n >= 0.0)
   {
      t = n;
      color->r = (float)(0.020 + 0.080 * t);
      color->g = (float)(0.070 + 0.860 * t);
      color->b = (float)(0.230 + 0.770 * t);
   }
   else
   {
      t = -n;
      color->r = (float)(0.180 + 0.820 * t);
      color->g = (float)(0.020 + 0.050 * t);
      color->b = (float)(0.100 + 0.580 * t);
   }
   ngClampColor(color);
}

static void ngWeightColor(float value, float maximumValue, NG_Color *color)
{
   double t;
   maximumValue = ngClampFloat(maximumValue, 0.0001f, 1000000.0f);
   t = ngClampFloat((float)(ngAbs(value) / maximumValue), 0.0f, 1.0f);
   if (value >= 0.0f)
   {
      color->r = (float)(0.030 + 0.060 * t);
      color->g = (float)(0.180 + 0.760 * t);
      color->b = (float)(0.420 + 0.580 * t);
   }
   else
   {
      color->r = (float)(0.380 + 0.620 * t);
      color->g = (float)(0.020 + 0.040 * t);
      color->b = (float)(0.180 + 0.600 * t);
   }
   ngClampColor(color);
}

static void ngEdgeGradientColor(
   float valueA,
   float valueB,
   float edgeWeight,
   double t,
   const NG_VisualMetrics *metrics,
   NG_Color *color)
{
   NG_Color colorA, colorB, stateColor, weightColor;
   double weightStrength, blend, flowStrength, brightness;

   ngActivationColor(valueA, metrics->maximumNodeValue, &colorA);
   ngActivationColor(valueB, metrics->maximumNodeValue, &colorB);
   stateColor = ngLerpColor(colorA, colorB, t);
   ngWeightColor(edgeWeight, metrics->maximumEdgeWeight, &weightColor);

   weightStrength = ngAbs(edgeWeight) / metrics->maximumEdgeWeight;
   blend = 0.14 + 0.20 * weightStrength;
   *color = ngAddColor(
      ngScaleColor(stateColor, 1.0 - blend),
      ngScaleColor(weightColor, blend));

   flowStrength = ngAbs(edgeWeight * (valueA - valueB)) /
      metrics->maximumFlow;
   brightness = 0.78 + 0.34 * ngClampFloat(
      (float)flowStrength, 0.0f, 1.0f);
   *color = ngScaleColor(*color, brightness);
   ngClampColor(color);
}

static void ngPulseColor(int phase, float flow, float maximumFlow, NG_Color *color)
{
   if (phase != NG_PHASE_DIFFUSION)
   {
      color->r = 0.015f;
      color->g = 0.025f;
      color->b = 0.055f;
      return;
   }
   ngWeightColor(flow, maximumFlow, color);
   *color = ngScaleColor(*color, 1.20);
   ngClampColor(color);
}

static SI_Error ngCreateAnimatedMaterial(
   const SAA_Scene *scene,
   const char *name,
   const NG_AnimationData *animation,
   const NG_Color *colors,
   SAA_Elem *material)
{
   SAA_Elem redFcurve, greenFcurve, blueFcurve;
   float *redValues, *greenValues, *blueValues;
   float strength;
   SI_Error result;
   int i;

   memset(&redFcurve, 0, sizeof(redFcurve));
   memset(&greenFcurve, 0, sizeof(greenFcurve));
   memset(&blueFcurve, 0, sizeof(blueFcurve));
   if (animation == NULL || colors == NULL || animation->keyCount <= 0)
      return SI_ERR_BAD_ARGUMENT;

   result = SAA_materialCreate(scene, material);
   if (result != SI_SUCCESS) return result;
   result = SAA_elementSetName(scene, material, name);
   if (result != SI_SUCCESS) goto fail;
   result = SAA_materialSetShadingModel(scene, material, SAA_SHM_BLINN);
   if (result != SI_SUCCESS) goto fail;
   result = SAA_materialSetAmbient(scene, material, 0.012f, 0.016f, 0.025f);
   if (result != SI_SUCCESS) goto fail;
   result = SAA_materialSetDiffuse(
      scene, material, colors[0].r, colors[0].g, colors[0].b);
   if (result != SI_SUCCESS) goto fail;

   strength = 0.0f;
   for (i = 0; i < animation->keyCount; ++i)
   {
      if (colors[i].r > strength) strength = colors[i].r;
      if (colors[i].g > strength) strength = colors[i].g;
      if (colors[i].b > strength) strength = colors[i].b;
   }
   strength = ngClampFloat(strength, 0.0f, 1.0f);
   result = SAA_materialSetSpecular(
      scene, material,
      0.62f + 0.28f * strength,
      0.66f + 0.26f * strength,
      0.76f + 0.22f * strength);
   if (result != SI_SUCCESS) goto fail;
   result = SAA_materialSetSpecularDecay(
      scene, material, 34.0f + 24.0f * strength);
   if (result != SI_SUCCESS) goto fail;
   result = SAA_materialSetReflection(
      scene, material, 0.04f + 0.10f * strength);
   if (result != SI_SUCCESS) goto fail;

   result = SAA_materialFcurveCreateDiffuse(
      scene, material, &redFcurve, &greenFcurve, &blueFcurve);
   if (result != SI_SUCCESS) goto fail;

   redValues = (float *)malloc(sizeof(float) * animation->keyCount);
   greenValues = (float *)malloc(sizeof(float) * animation->keyCount);
   blueValues = (float *)malloc(sizeof(float) * animation->keyCount);
   if (redValues == NULL || greenValues == NULL || blueValues == NULL)
   {
      free(redValues); free(greenValues); free(blueValues);
      result = SI_ERR_ALLOC_PROBLEM;
      goto fail;
   }
   for (i = 0; i < animation->keyCount; ++i)
   {
      redValues[i] = colors[i].r;
      greenValues[i] = colors[i].g;
      blueValues[i] = colors[i].b;
   }

   result = ngAddFcurveKeys(
      scene, &redFcurve, animation->keyCount, animation->times, redValues);
   if (result == SI_SUCCESS)
      result = ngAddFcurveKeys(
         scene, &greenFcurve, animation->keyCount, animation->times, greenValues);
   if (result == SI_SUCCESS)
      result = ngAddFcurveKeys(
         scene, &blueFcurve, animation->keyCount, animation->times, blueValues);
   free(redValues); free(greenValues); free(blueValues);
   if (result != SI_SUCCESS) goto fail;
   return SI_SUCCESS;

fail:
   SAA_elementDestroy(scene, material);
   return result;
}

static SI_Error ngAddFcurveKeys(
   const SAA_Scene *scene,
   const SAA_Elem *fcurve,
   int keyCount,
   const float *times,
   const float *values)
{
   SAA_SubElem *keys;
   SI_Error result;

   keys = (SAA_SubElem *)calloc(keyCount, sizeof(SAA_SubElem));
   if (keys == NULL) return SI_ERR_ALLOC_PROBLEM;
   result = SAA_fcurveKeyCreate(scene, fcurve, keyCount, keys);
   if (result == SI_SUCCESS)
      result = SAA_fcurveKeySetTime(scene, fcurve, keyCount, keys, times);
   if (result == SI_SUCCESS)
      result = SAA_fcurveKeySetValue(scene, fcurve, keyCount, keys, values);
   if (result == SI_SUCCESS)
      result = SAA_fcurveSetInterpolation(scene, fcurve, SAA_INT_LINEAR);
   free(keys);
   return result;
}

static SI_Error ngAssignGlobalMaterial(
   const SAA_Scene *scene,
   const SAA_Elem *model,
   const SAA_Elem *material)
{
   SAA_SubElem *polygons;
   SAA_Boolean *previousSelection, *allSelected;
   SI_Error result, restoreResult;
   int polygonCount, i;

   polygons = NULL;
   previousSelection = NULL;
   allSelected = NULL;
   result = SAA_meshGetNbPolygons(scene, model, &polygonCount);
   if (result != SI_SUCCESS) return result;
   if (polygonCount <= 0) return SI_ERR_WRONG_COUNT;

   polygons = (SAA_SubElem *)malloc(sizeof(SAA_SubElem) * polygonCount);
   previousSelection = (SAA_Boolean *)malloc(sizeof(SAA_Boolean) * polygonCount);
   allSelected = (SAA_Boolean *)malloc(sizeof(SAA_Boolean) * polygonCount);
   if (polygons == NULL || previousSelection == NULL || allSelected == NULL)
   {
      free(polygons); free(previousSelection); free(allSelected);
      return SI_ERR_ALLOC_PROBLEM;
   }

   result = SAA_meshGetPolygons(
      scene, model, SAA_GEOM_ORIGINAL, 0, polygonCount, polygons);
   if (result != SI_SUCCESS) goto done;
   result = SAA_polygonGetSelected(
      scene, model, polygonCount, polygons, previousSelection);
   if (result != SI_SUCCESS) goto done;
   for (i = 0; i < polygonCount; ++i) allSelected[i] = TRUE;
   result = SAA_polygonSetSelected(
      scene, model, polygonCount, polygons, allSelected);
   if (result != SI_SUCCESS) goto done;
   result = SAA_modelRelationCreateMat(scene, model, material);
   restoreResult = SAA_polygonSetSelected(
      scene, model, polygonCount, polygons, previousSelection);
   if (result == SI_SUCCESS && restoreResult != SI_SUCCESS)
      result = restoreResult;

done:
   free(polygons); free(previousSelection); free(allSelected);
   return result;
}

static SI_Error ngMaterialListInit(NG_MaterialList *list, int capacity)
{
   memset(list, 0, sizeof(*list));
   if (capacity <= 0) capacity = 1;
   list->items = (SAA_Elem *)calloc(capacity, sizeof(SAA_Elem));
   if (list->items == NULL) return SI_ERR_ALLOC_PROBLEM;
   list->capacity = capacity;
   return SI_SUCCESS;
}

static SI_Error ngMaterialListAdd(NG_MaterialList *list, const SAA_Elem *material)
{
   if (list == NULL || material == NULL || list->count >= list->capacity)
      return SI_ERR_WRONG_COUNT;
   list->items[list->count++] = *material;
   return SI_SUCCESS;
}

static void ngMaterialListRelease(
   const SAA_Scene *scene,
   NG_MaterialList *list,
   SAA_Boolean destroySceneElements)
{
   int i;
   if (list == NULL) return;
   if (destroySceneElements)
      for (i = list->count - 1; i >= 0; --i)
         SAA_elementDestroy(scene, &list->items[i]);
   free(list->items);
   memset(list, 0, sizeof(*list));
}

/* ------------------------------------------------------------------------- */
/* Shape-key baking                                                          */
/* ------------------------------------------------------------------------- */

static SI_Error ngGetTemplateFromSphere(
   const SAA_Scene *scene,
   const SAA_Elem *sphere,
   NG_NodeTemplate *templ)
{
   SI_Error result;
   int i;

   memset(templ, 0, sizeof(*templ));
   result = SAA_modelGetNbVertices(scene, sphere, &templ->count);
   if (result != SI_SUCCESS) return result;
   templ->vertices = (SAA_DVector *)calloc(templ->count, sizeof(SAA_DVector));
   templ->unitDirections = (NG_Vector3 *)calloc(templ->count, sizeof(NG_Vector3));
   if (templ->vertices == NULL || templ->unitDirections == NULL)
   {
      ngFreeNodeTemplate(templ);
      return SI_ERR_ALLOC_PROBLEM;
   }
   result = SAA_modelGetVertices(
      scene, sphere, SAA_GEOM_ORIGINAL, 0, templ->count, templ->vertices);
   if (result != SI_SUCCESS)
   {
      ngFreeNodeTemplate(templ);
      return result;
   }
   for (i = 0; i < templ->count; ++i)
   {
      NG_Vector3 v;
      v.x = templ->vertices[i].x;
      v.y = templ->vertices[i].y;
      v.z = templ->vertices[i].z;
      templ->unitDirections[i] = ngNormalize(v);
   }
   return SI_SUCCESS;
}

static void ngFreeNodeTemplate(NG_NodeTemplate *templ)
{
   if (templ == NULL) return;
   free(templ->vertices);
   free(templ->unitDirections);
   memset(templ, 0, sizeof(*templ));
}

static void ngBuildNodeVertices(
   const NG_NodeTemplate *templ,
   NG_Vector3 center,
   double radius,
   SAA_DVector *vertices)
{
   int i;
   for (i = 0; i < templ->count; ++i)
   {
      vertices[i].x = center.x + templ->unitDirections[i].x * radius;
      vertices[i].y = center.y + templ->unitDirections[i].y * radius;
      vertices[i].z = center.z + templ->unitDirections[i].z * radius;
      vertices[i].w = 1.0;
   }
}

static SI_Error ngBakeNodeShapes(
   const NG_Parameters *parameters,
   const NG_TrainingResult *training,
   const NG_AnimationData *animation,
   const NG_NodeTemplate *templ,
   const SAA_Elem *nodeModel,
   int nodeIndex)
{
   SAA_DVector *vertices;
   SI_Error result;
   int key;

   vertices = (SAA_DVector *)malloc(sizeof(SAA_DVector) * templ->count);
   if (vertices == NULL) return SI_ERR_ALLOC_PROBLEM;
   result = SI_SUCCESS;

   for (key = 0; key < animation->keyCount && result == SI_SUCCESS; ++key)
   {
      NG_Vector3 position;
      double radius;
      int shapeId;
      ngBuildAnimatedNodePose(
         parameters, training,
         &animation->nodeValues[key * NG_MAX_NODES],
         &animation->edgeWeights[key * NG_MAX_EDGES],
         nodeIndex, &position, &radius);
      ngBuildNodeVertices(templ, position, radius, vertices);
      result = SAA_modelAddShape(
         &g_scene, nodeModel, animation->times[key],
         templ->count, vertices, &shapeId);
      if (result == SI_SUCCESS && key == 0)
         result = SAA_modelSetShapeAnimMode(&g_scene, nodeModel, SAA_ANIM_AVERAGE);
      if (result == SI_SUCCESS && key == 0)
         result = SAA_modelSetShapeInterpolation(&g_scene, nodeModel, SAA_ANIM_LINEAR);
   }
   free(vertices);
   return result;
}

static SI_Error ngBakeEdgeSegmentShapes(
   const NG_Parameters *parameters,
   const NG_TrainingResult *training,
   const NG_AnimationData *animation,
   const NG_VisualMetrics *metrics,
   const SAA_Elem *edgeModel,
   int edgeIndex,
   int segmentIndex)
{
   SAA_DVector vertices[NG_EDGE_SIDES * 2];
   SI_Error result;
   int key, a, b;

   a = training->edges[edgeIndex].a;
   b = training->edges[edgeIndex].b;
   result = SI_SUCCESS;
   for (key = 0; key < animation->keyCount && result == SI_SUCCESS; ++key)
   {
      NG_Vector3 positionA, positionB, start, end, segmentStart, segmentEnd;
      double radiusA, radiusB, edgeRadius;
      int shapeId;

      ngBuildAnimatedNodePose(
         parameters, training,
         &animation->nodeValues[key * NG_MAX_NODES],
         &animation->edgeWeights[key * NG_MAX_EDGES],
         a, &positionA, &radiusA);
      ngBuildAnimatedNodePose(
         parameters, training,
         &animation->nodeValues[key * NG_MAX_NODES],
         &animation->edgeWeights[key * NG_MAX_EDGES],
         b, &positionB, &radiusB);
      ngComputeShortenedEdge(
         positionA, positionB, radiusA, radiusB, &start, &end);
      ngComputeSegmentEndpoints(
         start, end, segmentIndex, parameters->gradientSegments,
         &segmentStart, &segmentEnd);
      edgeRadius = ngComputeEdgeRadius(
         parameters,
         animation->edgeWeights[key * NG_MAX_EDGES + edgeIndex],
         metrics->maximumEdgeWeight);
      ngBuildAnimatedEdgeVertices(segmentStart, segmentEnd, edgeRadius, vertices);
      result = SAA_modelAddShape(
         &g_scene, edgeModel, animation->times[key],
         NG_EDGE_SIDES * 2, vertices, &shapeId);
      if (result == SI_SUCCESS && key == 0)
         result = SAA_modelSetShapeAnimMode(&g_scene, edgeModel, SAA_ANIM_AVERAGE);
      if (result == SI_SUCCESS && key == 0)
         result = SAA_modelSetShapeInterpolation(&g_scene, edgeModel, SAA_ANIM_LINEAR);
   }
   return result;
}

/* ------------------------------------------------------------------------- */
/* Flow pulses                                                               */
/* ------------------------------------------------------------------------- */

static void ngBuildPulsePose(
   const NG_Parameters *parameters,
   const NG_TrainingResult *training,
   const NG_AnimationData *animation,
   const NG_VisualMetrics *metrics,
   int keyIndex,
   int edgeIndex,
   int pulseRank,
   NG_Vector3 *position,
   double *radius,
   float *flowOut)
{
   int a, b;
   NG_Vector3 positionA, positionB, start, end;
   double radiusA, radiusB;
   float valueA, valueB, weight, flow;

   a = training->edges[edgeIndex].a;
   b = training->edges[edgeIndex].b;
   valueA = animation->nodeValues[keyIndex * NG_MAX_NODES + a];
   valueB = animation->nodeValues[keyIndex * NG_MAX_NODES + b];
   weight = animation->edgeWeights[keyIndex * NG_MAX_EDGES + edgeIndex];
   ngBuildAnimatedNodePose(
      parameters, training,
      &animation->nodeValues[keyIndex * NG_MAX_NODES],
      &animation->edgeWeights[keyIndex * NG_MAX_EDGES],
      a, &positionA, &radiusA);
   ngBuildAnimatedNodePose(
      parameters, training,
      &animation->nodeValues[keyIndex * NG_MAX_NODES],
      &animation->edgeWeights[keyIndex * NG_MAX_EDGES],
      b, &positionB, &radiusB);
   ngComputeShortenedEdge(
      positionA, positionB, radiusA, radiusB, &start, &end);

   flow = (float)ngAbs(weight) * (valueA - valueB);
   *flowOut = flow;
   if (animation->phases[keyIndex] != NG_PHASE_DIFFUSION)
   {
      *position = ngLerpVector(start, end, 0.50);
      *radius = parameters->nodeRadius * 0.015;
   }
   else
   {
      double progress = (double)animation->phaseSteps[keyIndex] /
         (double)parameters->diffusionSteps;
      double t, normalizedFlow;
      progress = ngClampFloat((float)progress, 0.0f, 1.0f);
      t = 0.05 + 0.90 * progress;
      if (flow < 0.0f) t = 1.0 - t;
      t += 0.035 * (double)(pulseRank % 3 - 1);
      t = ngClampFloat((float)t, 0.03f, 0.97f);
      *position = ngLerpVector(start, end, t);
      normalizedFlow = ngAbs(flow) / metrics->maximumFlow;
      *radius = parameters->nodeRadius *
         (0.10 + 0.25 * sqrt(ngClampFloat(
            (float)normalizedFlow, 0.0f, 1.0f)));
   }
}

static SI_Error ngBakePulseShapes(
   const NG_Parameters *parameters,
   const NG_TrainingResult *training,
   const NG_AnimationData *animation,
   const NG_VisualMetrics *metrics,
   const NG_NodeTemplate *templ,
   const SAA_Elem *pulseModel,
   int edgeIndex,
   int pulseRank)
{
   SAA_DVector *vertices;
   SI_Error result;
   int key;

   vertices = (SAA_DVector *)malloc(sizeof(SAA_DVector) * templ->count);
   if (vertices == NULL) return SI_ERR_ALLOC_PROBLEM;
   result = SI_SUCCESS;
   for (key = 0; key < animation->keyCount && result == SI_SUCCESS; ++key)
   {
      NG_Vector3 position;
      double radius;
      float flow;
      int shapeId;
      ngBuildPulsePose(
         parameters, training, animation, metrics,
         key, edgeIndex, pulseRank, &position, &radius, &flow);
      ngBuildNodeVertices(templ, position, radius, vertices);
      result = SAA_modelAddShape(
         &g_scene, pulseModel, animation->times[key],
         templ->count, vertices, &shapeId);
      if (result == SI_SUCCESS && key == 0)
         result = SAA_modelSetShapeAnimMode(&g_scene, pulseModel, SAA_ANIM_AVERAGE);
      if (result == SI_SUCCESS && key == 0)
         result = SAA_modelSetShapeInterpolation(&g_scene, pulseModel, SAA_ANIM_LINEAR);
   }
   free(vertices);
   return result;
}

static int ngSelectStrongestEdges(
   const NG_Parameters *parameters,
   const NG_TrainingResult *training,
   int *edgeIndices)
{
   SAA_Boolean used[NG_MAX_EDGES];
   int requested, selected;

   memset(used, 0, sizeof(used));
   requested = ngClampInt(parameters->pulseCount, 0, training->edgeCount);
   selected = 0;
   while (selected < requested)
   {
      int best = -1;
      float bestWeight = -1.0f;
      int i;
      for (i = 0; i < training->edgeCount; ++i)
      {
         float weight;
         if (used[i]) continue;
         weight = (float)ngAbs(training->epochEdgeWeights[
            parameters->epochs * NG_MAX_EDGES + i]);
         if (weight > bestWeight)
         {
            bestWeight = weight;
            best = i;
         }
      }
      if (best < 0) break;
      used[best] = TRUE;
      edgeIndices[selected++] = best;
   }
   return selected;
}
/* ------------------------------------------------------------------------- */
/* Edge mesh                                                                 */
/* ------------------------------------------------------------------------- */

static SI_Error ngCreateEdgeMesh(
   const SAA_Scene *scene,
   NG_Vector3 start,
   NG_Vector3 end,
   double radius,
   SAA_Elem *mesh)
{
   SAA_DVector vertices[NG_EDGE_SIDES * 2];
   int nbCtrlVertices[NG_EDGE_SIDES];
   int ctrlVertexIndices[NG_EDGE_SIDES * 4];
   int polygon, cursor;
   SI_Error result;

   ngBuildAnimatedEdgeVertices(start, end, radius, vertices);
   cursor = 0;
   for (polygon = 0; polygon < NG_EDGE_SIDES; ++polygon)
   {
      int next = (polygon + 1) % NG_EDGE_SIDES;
      nbCtrlVertices[polygon] = 4;
      ctrlVertexIndices[cursor++] = polygon;
      ctrlVertexIndices[cursor++] = next;
      ctrlVertexIndices[cursor++] = NG_EDGE_SIDES + next;
      ctrlVertexIndices[cursor++] = NG_EDGE_SIDES + polygon;
   }

   result = SAA_meshCreate(
      scene, NG_EDGE_SIDES * 2, vertices,
      NG_EDGE_SIDES, nbCtrlVertices, ctrlVertexIndices, mesh);
   if (result != SI_SUCCESS) return result;
   result = ngPrepareMeshNormals(scene, mesh);
   if (result != SI_SUCCESS)
   {
      SAA_elementDestroy(scene, mesh);
      return result;
   }
   return SI_SUCCESS;
}

static SI_Error ngPrepareMeshNormals(
   const SAA_Scene *scene,
   const SAA_Elem *mesh)
{
   SI_Error result;
   result = SAA_meshSetNormalsFlag(scene, mesh, SAA_AUTOMATIC_DISCONTINUITY);
   if (result != SI_SUCCESS) return result;
   return SAA_meshComputeNormals(scene, mesh, SAA_GEOM_ORIGINAL, 0);
}

/* ------------------------------------------------------------------------- */
/* Cleanup                                                                   */
/* ------------------------------------------------------------------------- */

static void ngCleanupPartialGraph(
   const SAA_Scene *scene,
   SAA_Boolean rootCreated,
   SAA_Elem *root,
   NG_MaterialList *materials)
{
   if (rootCreated) SAA_modelDestroyBranch(scene, root);
   ngMaterialListRelease(scene, materials, TRUE);
}

static void ngFreeTrainingResult(NG_TrainingResult *training)
{
   if (training == NULL) return;
   free(training->epochNodeValues);
   free(training->epochEdgeWeights);
   free(training->epochLosses);
   memset(training, 0, sizeof(*training));
}

/* ------------------------------------------------------------------------- */
/* Math and random helpers                                                   */
/* ------------------------------------------------------------------------- */

static int ngClampInt(int value, int minimum, int maximum)
{
   if (value < minimum) return minimum;
   if (value > maximum) return maximum;
   return value;
}

static float ngClampFloat(float value, float minimum, float maximum)
{
   if (value < minimum) return minimum;
   if (value > maximum) return maximum;
   return value;
}

static double ngAbs(double value)
{
   return value < 0.0 ? -value : value;
}

static double ngMax(double a, double b)
{
   return a > b ? a : b;
}

static double ngLength(NG_Vector3 value)
{
   return sqrt(value.x * value.x + value.y * value.y + value.z * value.z);
}

static double ngDistanceSquared(NG_Vector3 a, NG_Vector3 b)
{
   NG_Vector3 difference = ngSubtract(a, b);
   return difference.x * difference.x +
          difference.y * difference.y +
          difference.z * difference.z;
}

static NG_Vector3 ngAdd(NG_Vector3 a, NG_Vector3 b)
{
   NG_Vector3 result;
   result.x = a.x + b.x;
   result.y = a.y + b.y;
   result.z = a.z + b.z;
   return result;
}

static NG_Vector3 ngSubtract(NG_Vector3 a, NG_Vector3 b)
{
   NG_Vector3 result;
   result.x = a.x - b.x;
   result.y = a.y - b.y;
   result.z = a.z - b.z;
   return result;
}

static NG_Vector3 ngScale(NG_Vector3 value, double scalar)
{
   NG_Vector3 result;
   result.x = value.x * scalar;
   result.y = value.y * scalar;
   result.z = value.z * scalar;
   return result;
}

static NG_Vector3 ngCross(NG_Vector3 a, NG_Vector3 b)
{
   NG_Vector3 result;
   result.x = a.y * b.z - a.z * b.y;
   result.y = a.z * b.x - a.x * b.z;
   result.z = a.x * b.y - a.y * b.x;
   return result;
}

static NG_Vector3 ngNormalize(NG_Vector3 value)
{
   double length = ngLength(value);
   if (length < NG_EPSILON)
   {
      value.x = 1.0; value.y = 0.0; value.z = 0.0;
      return value;
   }
   return ngScale(value, 1.0 / length);
}

static NG_Vector3 ngLerpVector(NG_Vector3 a, NG_Vector3 b, double t)
{
   return ngAdd(ngScale(a, 1.0 - t), ngScale(b, t));
}

static NG_Color ngLerpColor(NG_Color a, NG_Color b, double t)
{
   NG_Color result;
   result.r = (float)((1.0 - t) * a.r + t * b.r);
   result.g = (float)((1.0 - t) * a.g + t * b.g);
   result.b = (float)((1.0 - t) * a.b + t * b.b);
   return result;
}

static NG_Color ngScaleColor(NG_Color color, double factor)
{
   color.r = (float)(color.r * factor);
   color.g = (float)(color.g * factor);
   color.b = (float)(color.b * factor);
   return color;
}

static NG_Color ngAddColor(NG_Color a, NG_Color b)
{
   NG_Color result;
   result.r = a.r + b.r;
   result.g = a.g + b.g;
   result.b = a.b + b.b;
   return result;
}

static void ngClampColor(NG_Color *color)
{
   color->r = ngClampFloat(color->r, 0.0f, 1.0f);
   color->g = ngClampFloat(color->g, 0.0f, 1.0f);
   color->b = ngClampFloat(color->b, 0.0f, 1.0f);
}

static unsigned long ngRandomNext(NG_Random *random)
{
   random->state = random->state * 1664525UL + 1013904223UL;
   return random->state;
}

static float ngRandomUnit(NG_Random *random)
{
   unsigned long value = ngRandomNext(random) & 0x00FFFFFFUL;
   return (float)value / 16777215.0f;
}

