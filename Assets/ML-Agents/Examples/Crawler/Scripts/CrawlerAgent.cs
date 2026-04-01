using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgentsExamples;
using Unity.MLAgents.Sensors;
using Random = UnityEngine.Random;

[RequireComponent(typeof(JointDriveController))] // Required to set joint forces
public class CrawlerAgent : Agent
{

    [Header("Walk Speed")]
    [Range(0.1f, m_maxWalkingSpeed)]
    [SerializeField]
    [Tooltip(
        "The speed the agent will try to match.\n\n" +
        "TRAINING:\n" +
        "For VariableSpeed envs, this value will randomize at the start of each training episode.\n" +
        "Otherwise the agent will try to match the speed set here.\n\n" +
        "INFERENCE:\n" +
        "During inference, VariableSpeed agents will modify their behavior based on this value " +
        "whereas the CrawlerDynamic & CrawlerStatic agents will run at the speed specified during training "
    )]
    //The walking speed to try and achieve
    private float m_TargetWalkingSpeed = m_maxWalkingSpeed;

    const float m_maxWalkingSpeed = 15; //The max walking speed

    Rigidbody footRigidbody;


    //The current target walking speed. Clamped because a value of zero will cause NaNs
    public float TargetWalkingSpeed
    {
        get { return m_TargetWalkingSpeed; }
        set { m_TargetWalkingSpeed = Mathf.Clamp(value, .1f, m_maxWalkingSpeed); }
    }

    //The direction an agent will walk during training.
    [Header("Target To Walk Towards")]
    public Transform TargetPrefab; //Target prefab to use in Dynamic envs
    private Transform m_Target; //Target the agent will walk towards during training.

    [Header("Body Parts")][Space(10)] public Transform body;
    public Transform leg0Upper;
    public Transform leg0Mid;
    public Transform leg0Lower;

    public Transform leg1Upper;
    public Transform leg1Mid;
    public Transform leg1Lower;

    public Transform leg2Upper;
    public Transform leg2Mid;
    public Transform leg2Lower;

    public Transform leg3Upper;
    public Transform leg3Mid;
    public Transform leg3Lower;

    public Transform leg4Upper;
    public Transform leg4Mid;
    public Transform leg4Lower;

    public Transform leg5Upper;
    public Transform leg5Mid;
    public Transform leg5Lower;

    public Transform leg6Upper;
    public Transform leg6Mid;
    public Transform leg6Lower;

    public Transform leg7Upper;
    public Transform leg7Mid;
    public Transform leg7Lower;



    //This will be used as a stabilized model space reference point for observations
    //Because ragdolls can move erratically during training, using a stabilized reference transform improves learning
    OrientationCubeController m_OrientationCube;

    //The indicator graphic gameobject that points towards the target
    DirectionIndicator m_DirectionIndicator;
    JointDriveController m_JdController;

    [Header("Foot Grounded Visualization")]
    [Space(10)]
    public bool useFootGroundedVisualization;

    public MeshRenderer foot0;
    public MeshRenderer foot1;
    public MeshRenderer foot2;
    public MeshRenderer foot3;
    public MeshRenderer foot4;
    public MeshRenderer foot5;
    public MeshRenderer foot6;
    public MeshRenderer foot7;
    public Material groundedMaterial;
    public Material unGroundedMaterial;

    public override void Initialize()
    {
        SpawnTarget(TargetPrefab, transform.position); // spawn target

        m_OrientationCube = GetComponentInChildren<OrientationCubeController>();
        m_DirectionIndicator = GetComponentInChildren<DirectionIndicator>();
        m_JdController = GetComponent<JointDriveController>();

        // Setup each body part
        m_JdController.SetupBodyPart(body);
        m_JdController.SetupBodyPart(leg0Upper);
        m_JdController.SetupBodyPart(leg0Mid);
        m_JdController.SetupBodyPart(leg0Lower);

        m_JdController.SetupBodyPart(leg1Upper);
        m_JdController.SetupBodyPart(leg1Mid);
        m_JdController.SetupBodyPart(leg1Lower);

        m_JdController.SetupBodyPart(leg2Upper);
        m_JdController.SetupBodyPart(leg2Mid);
        m_JdController.SetupBodyPart(leg2Lower);

        m_JdController.SetupBodyPart(leg3Upper);
        m_JdController.SetupBodyPart(leg3Mid);
        m_JdController.SetupBodyPart(leg3Lower);

        m_JdController.SetupBodyPart(leg4Upper);
        m_JdController.SetupBodyPart(leg4Mid);
        m_JdController.SetupBodyPart(leg4Lower);

        m_JdController.SetupBodyPart(leg5Upper);
        m_JdController.SetupBodyPart(leg5Mid);
        m_JdController.SetupBodyPart(leg5Lower);

        m_JdController.SetupBodyPart(leg6Upper);
        m_JdController.SetupBodyPart(leg6Mid);
        m_JdController.SetupBodyPart(leg6Lower);

        m_JdController.SetupBodyPart(leg7Upper);
        m_JdController.SetupBodyPart(leg7Mid);
        m_JdController.SetupBodyPart(leg7Lower);


    }

    /// <summary>
    /// Spawns a target prefab at pos
    /// </summary>
    /// <param name="prefab"></param>
    /// <param name="pos"></param>
    void SpawnTarget(Transform prefab, Vector3 pos)
    {
        m_Target = Instantiate(prefab, pos, Quaternion.identity, transform.parent);
    }

    /// <summary>
    /// Loop over body parts and reset them to initial conditions.
    /// </summary>
    public override void OnEpisodeBegin()
    {
        foreach (var bodyPart in m_JdController.bodyPartsDict.Values)
        {
            bodyPart.Reset(bodyPart);
        }

        //Random start rotation to help generalize
        body.rotation = Quaternion.Euler(0, Random.Range(0.0f, 360.0f), 0);

        UpdateOrientationObjects();

        //Set our goal walking speed
        TargetWalkingSpeed = Random.Range(0.1f, m_maxWalkingSpeed);
    }

    /// <summary>
    /// Add relevant information on each body part to observations.
    /// </summary>
    public void CollectObservationBodyPart(BodyPart bp, VectorSensor sensor)
    {
        //GROUND CHECK
        sensor.AddObservation(bp.groundContact.touchingGround); // Is this bp touching the ground

        if (bp.rb.transform != body)
        {
            sensor.AddObservation(bp.currentStrength / m_JdController.maxJointForceLimit);
        }
    }

    /// <summary>
    /// Loop over body parts to add them to observation.
    /// </summary>
    public override void CollectObservations(VectorSensor sensor)
    {
        var cubeForward = m_OrientationCube.transform.forward;

        //velocity we want to match
        var velGoal = cubeForward * TargetWalkingSpeed;
        //ragdoll's avg vel
        var avgVel = GetAvgVelocity();

        //current ragdoll velocity. normalized
        sensor.AddObservation(Vector3.Distance(velGoal, avgVel));
        //avg body vel relative to cube
        sensor.AddObservation(m_OrientationCube.transform.InverseTransformDirection(avgVel));
        //vel goal relative to cube
        sensor.AddObservation(m_OrientationCube.transform.InverseTransformDirection(velGoal));
        //rotation delta
        sensor.AddObservation(Quaternion.FromToRotation(body.forward, cubeForward));

        //Add pos of target relative to orientation cube
        sensor.AddObservation(m_OrientationCube.transform.InverseTransformPoint(m_Target.transform.position));

        RaycastHit hit;
        float maxRaycastDist = 10;
        if (Physics.Raycast(body.position, Vector3.down, out hit, maxRaycastDist))
        {
            sensor.AddObservation(hit.distance / maxRaycastDist);
        }
        else
            sensor.AddObservation(1);

        foreach (var bodyPart in m_JdController.bodyPartsList)
        {
            CollectObservationBodyPart(bodyPart, sensor);
        }
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        // The dictionary with all the body parts in it are in the jdController
        var bpDict = m_JdController.bodyPartsDict;
            
        var continuousActions = actionBuffers.ContinuousActions;

        var discreteActions = actionBuffers.DiscreteActions;



        var i = -1;

        // Pick a new target joint rotation for each leg
        bpDict[leg0Upper].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], continuousActions[++i]);
        bpDict[leg1Upper].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], continuousActions[++i]);
        bpDict[leg2Upper].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], continuousActions[++i]);
        bpDict[leg3Upper].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], continuousActions[++i]);
        bpDict[leg4Upper].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], continuousActions[++i]);
        bpDict[leg5Upper].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], continuousActions[++i]);
        bpDict[leg6Upper].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], continuousActions[++i]); // New leg
        bpDict[leg7Upper].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], continuousActions[++i]); // New leg

        bpDict[leg0Mid].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], continuousActions[++i]);
        bpDict[leg1Mid].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], continuousActions[++i]);
        bpDict[leg2Mid].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], continuousActions[++i]);
        bpDict[leg3Mid].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], continuousActions[++i]);
        bpDict[leg4Mid].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], continuousActions[++i]);
        bpDict[leg5Mid].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], continuousActions[++i]);
        bpDict[leg6Mid].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], continuousActions[++i]); // New leg
        bpDict[leg7Mid].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], continuousActions[++i]); // New leg

        bpDict[leg0Lower].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], continuousActions[++i]);
        bpDict[leg1Lower].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], continuousActions[++i]);
        bpDict[leg2Lower].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], continuousActions[++i]);
        bpDict[leg3Lower].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], continuousActions[++i]);
        bpDict[leg4Lower].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], continuousActions[++i]);
        bpDict[leg5Lower].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], continuousActions[++i]);
        bpDict[leg6Lower].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], continuousActions[++i]);
        bpDict[leg7Lower].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], continuousActions[++i]);

      

        // Update joint strength for each leg
        bpDict[leg0Upper].SetJointStrength(continuousActions[++i]);
        bpDict[leg1Upper].SetJointStrength(continuousActions[++i]);
        bpDict[leg2Upper].SetJointStrength(continuousActions[++i]);
        bpDict[leg3Upper].SetJointStrength(continuousActions[++i]);
        bpDict[leg4Upper].SetJointStrength(continuousActions[++i]);
        bpDict[leg5Upper].SetJointStrength(continuousActions[++i]);
        bpDict[leg6Upper].SetJointStrength(continuousActions[++i]); // New leg
        bpDict[leg7Upper].SetJointStrength(continuousActions[++i]); // New leg

        bpDict[leg0Mid].SetJointStrength(continuousActions[++i]);
        bpDict[leg1Mid].SetJointStrength(continuousActions[++i]);
        bpDict[leg2Mid].SetJointStrength(continuousActions[++i]);
        bpDict[leg3Mid].SetJointStrength(continuousActions[++i]);
        bpDict[leg4Mid].SetJointStrength(continuousActions[++i]);
        bpDict[leg5Mid].SetJointStrength(continuousActions[++i]);
        bpDict[leg6Mid].SetJointStrength(continuousActions[++i]); // New leg
        bpDict[leg7Mid].SetJointStrength(continuousActions[++i]); // New leg

        bpDict[leg0Lower].SetJointStrength(continuousActions[++i]);
        bpDict[leg1Lower].SetJointStrength(continuousActions[++i]);
        bpDict[leg2Lower].SetJointStrength(continuousActions[++i]);
        bpDict[leg3Lower].SetJointStrength(continuousActions[++i]);
        bpDict[leg4Lower].SetJointStrength(continuousActions[++i]);
        bpDict[leg5Lower].SetJointStrength(continuousActions[++i]);
        bpDict[leg6Lower].SetJointStrength(continuousActions[++i]); // New leg
        bpDict[leg7Lower].SetJointStrength(continuousActions[++i]); // New leg



        /*  // Loop through all feet (0-7)
          int j = 0;

          bool freezeFootRequested = discreteActions[j++] == 1; // Assuming 1 means freeze

          Rigidbody footRigidbody = feet0.GetComponent<Rigidbody>();

          if (freezeFootRequested && bpDict[feet0].groundContact.touchingGround)
          {
              footRigidbody.constraints = RigidbodyConstraints.FreezePositionX |
                                          RigidbodyConstraints.FreezePositionY |
                                          RigidbodyConstraints.FreezePositionZ;
              foot0.material = groundedMaterial;
          }
          else
          {
              // Unfreeze foot if requested to move
              footRigidbody.constraints = RigidbodyConstraints.None;
              foot0.material = unGroundedMaterial;
          }

          */



    }


 



    void FixedUpdate()
    {
        UpdateOrientationObjects();

        // If enabled the feet will light up green when the foot is grounded.
        // This is just a visualization and isn't necessary for function
        
           foot0.material = m_JdController.bodyPartsDict[leg0Lower].groundContact.touchingGround
                ? groundedMaterial
                : unGroundedMaterial;
            foot1.material = m_JdController.bodyPartsDict[leg1Lower].groundContact.touchingGround
                ? groundedMaterial
                : unGroundedMaterial;
            foot2.material = m_JdController.bodyPartsDict[leg2Lower].groundContact.touchingGround
                ? groundedMaterial
                : unGroundedMaterial;
            foot3.material = m_JdController.bodyPartsDict[leg3Lower].groundContact.touchingGround
                ? groundedMaterial
                : unGroundedMaterial;
            foot4.material = m_JdController.bodyPartsDict[leg4Lower].groundContact.touchingGround
                ? groundedMaterial
                : unGroundedMaterial;
            foot5.material = m_JdController.bodyPartsDict[leg5Lower].groundContact.touchingGround
                ? groundedMaterial
                : unGroundedMaterial;
            foot6.material = m_JdController.bodyPartsDict[leg6Lower].groundContact.touchingGround
                ? groundedMaterial
                : unGroundedMaterial; // New foot for leg6
            foot7.material = m_JdController.bodyPartsDict[leg7Lower].groundContact.touchingGround
                ? groundedMaterial
                : unGroundedMaterial; // New foot for leg7
        

        var cubeForward = m_OrientationCube.transform.forward;

        // Set reward for this step according to mixture of the following elements.
        // a. Match target speed
        //This reward will approach 1 if it matches perfectly and approach zero as it deviates
        var matchSpeedReward = GetMatchingVelocityReward(cubeForward * TargetWalkingSpeed, GetAvgVelocity());

        // b. Rotation alignment with target direction.
        //This reward will approach 1 if it faces the target direction perfectly and approach zero as it deviates
        var lookAtTargetReward = (Vector3.Dot(cubeForward, body.forward) + 1) * .5F;

        AddReward(matchSpeedReward * lookAtTargetReward);
    }



    /// <summary>
    /// Update OrientationCube and DirectionIndicator
    /// </summary>
    void UpdateOrientationObjects()
    {
        m_OrientationCube.UpdateOrientation(body, m_Target);
        if (m_DirectionIndicator)
        {
            m_DirectionIndicator.MatchOrientation(m_OrientationCube.transform);
        }
    }

    /// <summary>
    ///Returns the average velocity of all of the body parts
    ///Using the velocity of the body only has shown to result in more erratic movement from the limbs
    ///Using the average helps prevent this erratic movement
    /// </summary>
    Vector3 GetAvgVelocity()
    {
        Vector3 velSum = Vector3.zero;
        Vector3 avgVel = Vector3.zero;

        //ALL RBS
        int numOfRb = 0;
        foreach (var item in m_JdController.bodyPartsList)
        {
            numOfRb++;
            velSum += item.rb.linearVelocity;
        }

        avgVel = velSum / numOfRb;
        return avgVel;
    }

    /// <summary>
    /// Normalized value of the difference in actual speed vs goal walking speed.
    /// </summary>
    public float GetMatchingVelocityReward(Vector3 velocityGoal, Vector3 actualVelocity)
    {
        //distance between our actual velocity and goal velocity
        var velDeltaMagnitude = Mathf.Clamp(Vector3.Distance(actualVelocity, velocityGoal), 0, TargetWalkingSpeed);

        //return the value on a declining sigmoid shaped curve that decays from 1 to 0
        //This reward will approach 1 if it matches perfectly and approach zero as it deviates
        return Mathf.Pow(1 - Mathf.Pow(velDeltaMagnitude / TargetWalkingSpeed, 2), 2);
    }

    /// <summary>
    /// Agent touched the target
    /// </summary>
    public void TouchedTarget()
    {
        AddReward(1f);
    }
}
