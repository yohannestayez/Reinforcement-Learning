using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

public class NavigationAgent : Agent
{
    // References
    public GameObject target;
    public Camera mainCamera;  // Reference to main camera
    
    // Environment parameters (matching simple environment)
    private const float MIN_POSITION = -8f;
    private float MAX_POSITION = 8f;
    private const float ARENA_SIZE = 20f; // -10 to 10, matching simple env
    private const int MAX_STEPS = 100;
    private int currentStep;
    private const float TARGET_REACH_DISTANCE = 0.5f;
    private const float MIN_SPAWN_DISTANCE = 2.0f;
    
    // Grid parameters (exactly matching simple environment)
    private const int GRID_LINES = 21;
    private const float GRID_SPACING = 1f;
    private GameObject[] gridLines;
    
    // Visualization parameters (matching simple environment pixel-perfect)
    private const int WINDOW_SIZE = 800;
    private const int PIXELS_PER_UNIT = 40; // Simple env uses 40 pixels per unit
    private const int CIRCLE_RADIUS_PIXELS = 10;
    
    void Start()
    {
        // Set up camera for pixel-perfect match
        if (mainCamera == null)
        {
            mainCamera = Camera.main;
        }
        SetupCamera();
        
        // Create materials with exact colors from simple environment
        Material agentMaterial = new Material(Shader.Find("Standard"));
        agentMaterial.color = new Color(0f, 0f, 1f, 1f); // RGB(0, 0, 255)
        
        Material targetMaterial = new Material(Shader.Find("Standard"));
        targetMaterial.color = new Color(1f, 0f, 0f, 1f); // RGB(255, 0, 0)
        
        Material gridMaterial = new Material(Shader.Find("Standard"));
        gridMaterial.color = new Color(50f/255f, 50f/255f, 50f/255f, 1f); // RGB(50, 50, 50)
        
        // Set materials
        GetComponent<Renderer>().material = agentMaterial;
        target.GetComponent<Renderer>().material = targetMaterial;
        
        // Set sizes to match simple environment circle sizes exactly
        float worldSpaceRadius = (float)CIRCLE_RADIUS_PIXELS / PIXELS_PER_UNIT;
        transform.localScale = new Vector3(worldSpaceRadius * 2, worldSpaceRadius * 2, worldSpaceRadius * 2);
        target.transform.localScale = new Vector3(worldSpaceRadius * 2, worldSpaceRadius * 2, worldSpaceRadius * 2);
        
        // Create grid
        CreateGrid(gridMaterial);
    }
    
    private void SetupCamera()
    {
        // Position camera for pixel-perfect top-down view
        mainCamera.transform.position = new Vector3(0, 20, 0);
        mainCamera.transform.rotation = Quaternion.Euler(90, 0, 0);
        mainCamera.orthographic = true;
        
        // Set size to match 800x800 window exactly
        // In Unity, orthographicSize is half the height in world units
        mainCamera.orthographicSize = WINDOW_SIZE / (2f * PIXELS_PER_UNIT);
        
        // Force resolution to match simple environment
        Screen.SetResolution(WINDOW_SIZE, WINDOW_SIZE, false);
        
        // Set background color to match
        mainCamera.backgroundColor = Color.black;
    }
    
    private void CreateGrid(Material gridMaterial)
    {
        gridLines = new GameObject[GRID_LINES * 2];
        GameObject gridParent = new GameObject("Grid");
        gridParent.transform.position = Vector3.zero;
        
        // Calculate line width to match 1-pixel width
        float lineWidth = 1f / PIXELS_PER_UNIT;
        
        for (int i = 0; i < GRID_LINES; i++)
        {
            float position = (i - GRID_LINES/2) * GRID_SPACING;
            
            // Create vertical line (1 pixel wide)
            GameObject verticalLine = GameObject.CreatePrimitive(PrimitiveType.Cube);
            verticalLine.transform.parent = gridParent.transform;
            verticalLine.transform.localScale = new Vector3(lineWidth, lineWidth, ARENA_SIZE);
            verticalLine.transform.position = new Vector3(position, 0.01f, 0f);
            verticalLine.GetComponent<Renderer>().material = gridMaterial;
            gridLines[i] = verticalLine;
            
            // Create horizontal line (1 pixel wide)
            GameObject horizontalLine = GameObject.CreatePrimitive(PrimitiveType.Cube);
            horizontalLine.transform.parent = gridParent.transform;
            horizontalLine.transform.localScale = new Vector3(ARENA_SIZE, lineWidth, lineWidth);
            horizontalLine.transform.position = new Vector3(0f, 0.01f, position);
            horizontalLine.GetComponent<Renderer>().material = gridMaterial;
            gridLines[i + GRID_LINES] = horizontalLine;
        }
    }
    
    public override void OnEpisodeBegin()
    {
        currentStep = 0;
        
        // Random initial position for agent (matching simple environment)
        float agentX = Random.Range(MIN_POSITION, MAX_POSITION);
        float agentZ = Random.Range(MIN_POSITION, MAX_POSITION);
        transform.position = new Vector3(agentX, 0.5f, agentZ);
        
        // Random target position
        float targetX = Random.Range(MIN_POSITION, MAX_POSITION);
        float targetZ = Random.Range(MIN_POSITION, MAX_POSITION);
        target.transform.position = new Vector3(targetX, 0.5f, targetZ);
        
        // Ensure target is not too close (matching simple environment minimum distance)
        while (Vector3.Distance(transform.position, target.transform.position) < MIN_SPAWN_DISTANCE)
        {
            targetX = Random.Range(MIN_POSITION, MAX_POSITION);
            targetZ = Random.Range(MIN_POSITION, MAX_POSITION);
            target.transform.position = new Vector3(targetX, 0.5f, targetZ);
        }
    }
    
    public override void CollectObservations(VectorSensor sensor)
    {
        // Collect observations (matching simple environment)
        sensor.AddObservation(transform.position.x); // Agent X
        sensor.AddObservation(transform.position.z); // Agent Z (Y in 2D)
        sensor.AddObservation(target.transform.position.x); // Target X
        sensor.AddObservation(target.transform.position.z); // Target Z (Y in 2D)
    }
    
    public override void OnActionReceived(ActionBuffers actions)
    {
        currentStep++;
        
        // Get continuous actions (matching simple environment -1 to 1 range)
        float moveX = actions.ContinuousActions[0];
        float moveZ = actions.ContinuousActions[1];
        
        // Move agent (matching simple environment clipping)
        float newX = Mathf.Clamp(transform.position.x + moveX, -10f, 10f);
        float newZ = Mathf.Clamp(transform.position.z + moveZ, -10f, 10f);
        transform.position = new Vector3(newX, 0.5f, newZ);
        
        // Calculate distance to target
        float distance = Vector3.Distance(new Vector3(transform.position.x, 0, transform.position.z),
                                       new Vector3(target.transform.position.x, 0, target.transform.position.z));
        
        // Calculate reward (matching simple environment)
        if (distance < TARGET_REACH_DISTANCE)
        {
            SetReward(1.0f);
            EndEpisode();
        }
        else
        {
            SetReward(-0.01f); // Small negative reward per step
            if (currentStep >= MAX_STEPS)
            {
                EndEpisode();
            }
        }
    }
    
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActionsOut = actionsOut.ContinuousActions;
        continuousActionsOut[0] = Input.GetAxisRaw("Horizontal");
        continuousActionsOut[1] = Input.GetAxisRaw("Vertical");
    }
}
