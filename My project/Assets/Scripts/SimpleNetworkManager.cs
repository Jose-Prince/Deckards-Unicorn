using UnityEngine;
using UnityEngine.UI;
using TMPro;
using Unity.Netcode;
using Unity.Netcode.Transports.UTP;
using UnityEngine.SceneManagement;
using System.Collections.Generic;
using Unity.Services.Core;
using Unity.Services.Authentication;
using Unity.Services.Relay;
using Unity.Services.Relay.Models;

public class SimpleNetworkManager : NetworkBehaviour
{
    [Header("UI References - Connection")]
    [SerializeField] private Button hostButton;
    [SerializeField] private Button joinButton;
    [SerializeField] private TMP_InputField ipAddressInput;
    [SerializeField] private TMP_Text codeRoomText;
    [SerializeField] private Button copyButton;
    [SerializeField] private TMP_Text joinLabel;

    [Header("UI References - Player Name")]
    [SerializeField] private TMP_InputField nameInput;
    [SerializeField] private Button nameButton;

    [Header("UI References - Lobby")]
    [SerializeField] private Transform playersListContent;
    [SerializeField] private GameObject playerListItemPrefab;
    [SerializeField] private Button startChatButton;

    [Header("Settings")]
    [SerializeField] private string chatSceneName = "ChatScene";

    [Header("PageButtons")]
    [SerializeField] private Button returnButton;
    [SerializeField] private Button leaveButton;

    private string playerName = "Player";
    private string currentRoomCode;
    private NetworkManager networkManager;
    private NetworkList<PlayerData> players = new NetworkList<PlayerData>();
    private UnityTransport transport;
    private Dictionary<ulong, GameObject> playerListItems = new();

    private void Awake()
    {
        DontDestroyOnLoad(gameObject);
    }

    private async void Start()
    {
        networkManager = NetworkManager.Singleton;
        transport = networkManager.GetComponent<UnityTransport>();

        await UnityServices.InitializeAsync();
        if (!AuthenticationService.Instance.IsSignedIn)
        {
            await AuthenticationService.Instance.SignInAnonymouslyAsync();
        }

        hostButton.onClick.AddListener(StartHost);
        joinButton.onClick.AddListener(StartClient);
        startChatButton.onClick.AddListener(LoadChatScene);

        networkManager.OnClientConnectedCallback += OnClientConnected;
        networkManager.OnClientDisconnectCallback += OnClientDisconnected;

        returnButton.onClick.AddListener(() => SceneManager.LoadScene("MainMenu"));
        leaveButton.onClick.AddListener(LeaveGame);
        leaveButton.gameObject.SetActive(false);
        codeRoomText.gameObject.SetActive(false);
        copyButton.gameObject.SetActive(false);

        nameButton.onClick.AddListener(SetPlayerName);
    }

    public override void OnNetworkSpawn()
    {
        if (IsServer)
        {
            players.OnListChanged += OnPlayersListChanged;
        }
    }

    private async void StartHost()
    {
        try
        {
            Allocation allocation = await RelayService.Instance.CreateAllocationAsync(8); // Max players

            string joinCode = await RelayService.Instance.GetJoinCodeAsync(allocation.AllocationId);
            currentRoomCode = joinCode;

            codeRoomText.text = $"Room Code: {joinCode}";
            codeRoomText.gameObject.SetActive(true);
            copyButton.gameObject.SetActive(true);

            copyButton.onClick.RemoveAllListeners();
            copyButton.onClick.AddListener(CopyRoomCodeToClipboard);

            transport.SetRelayServerData(
                allocation.RelayServer.IpV4,
                (ushort)allocation.RelayServer.Port,
                allocation.AllocationIdBytes,
                allocation.Key,
                allocation.ConnectionData
            );

            if (networkManager.StartHost())
            {
                ShowConnectedPanel();
                AddPlayerToList(networkManager.LocalClientId, $"{playerName} (Host)");
            }

            joinLabel.gameObject.SetActive(false);
        }
        catch (RelayServiceException e)
        {
            Debug.LogError(e);
        }
    }

    private async void StartClient()
    {
        string joinCode = ipAddressInput.text.Trim();

        if (string.IsNullOrEmpty(joinCode))
        {
            return;
        }

        try
        {
            JoinAllocation joinAlloc = await RelayService.Instance.JoinAllocationAsync(joinCode);

            transport.SetRelayServerData(
                joinAlloc.RelayServer.IpV4,
                (ushort)joinAlloc.RelayServer.Port,
                joinAlloc.AllocationIdBytes,
                joinAlloc.Key,
                joinAlloc.ConnectionData,
                joinAlloc.HostConnectionData
            );

            if (networkManager.StartClient())
            {
                Debug.Log($"Joined room {joinCode}");
            }
        }
        catch (RelayServiceException e)
        {
            Debug.LogError(e);
        }
    }

    private void OnClientConnected(ulong clientId)
    {
        if (IsServer)
        {
            string nameToAdd = clientId == NetworkManager.Singleton.LocalClientId
                ? $"{playerName} (Host)"
                : $"Player {clientId}";

            players.Add(new PlayerData
            {
                clientId = clientId,
                playerName = nameToAdd
            });
        }
    }

    private void OnClientDisconnected(ulong clientId)
    {
        if (IsServer)
        {
            for (int i = 0; i < players.Count; i++)
            {
                if (players[i].clientId == clientId)
                {
                    players.RemoveAt(i);
                    break;
                }
            }
        }
    }


    private void ShowConnectedPanel()
    {
        hostButton.gameObject.SetActive(false);
        joinButton.gameObject.SetActive(false);
        ipAddressInput.gameObject.SetActive(false);
        leaveButton.gameObject.SetActive(true);

        startChatButton.gameObject.SetActive(networkManager.IsHost);
    }

    private void HideConnectedPanel()
    {
        hostButton.gameObject.SetActive(true);
        joinButton.gameObject.SetActive(true);
        ipAddressInput.gameObject.SetActive(true);
        leaveButton.gameObject.SetActive(false);

        foreach (var item in playerListItems.Values)
        {
            Destroy(item);
        }
        playerListItems.Clear();
    }

    private void AddPlayerToList(ulong clientId, string playerName)
    {
        if (playerListItems.ContainsKey(clientId)) return;

        GameObject listItem = Instantiate(playerListItemPrefab, playersListContent);
        TMP_Text text = listItem.GetComponentInChildren<TMP_Text>();
        if (text != null) text.text = $" {playerName}";

        playerListItems.Add(clientId, listItem);
    }

    private void RemovePlayerFromList(ulong clientId)
    {
        if (playerListItems.ContainsKey(clientId))
        {
            Destroy(playerListItems[clientId]);
            playerListItems.Remove(clientId);
        }
    }

    private void LoadChatScene()
    {
        if (networkManager.IsHost)
        {
            networkManager.SceneManager.LoadScene(chatSceneName, LoadSceneMode.Single);
        }
    }

    private void OnDestroy()
    {
        if (networkManager != null)
        {
            networkManager.OnClientConnectedCallback -= OnClientConnected;
            networkManager.OnClientDisconnectCallback -= OnClientDisconnected;
        }
    }

    private void LeaveGame()
    {
        codeRoomText.gameObject.SetActive(false);
        copyButton.gameObject.SetActive(false);
        ipAddressInput.gameObject.SetActive(true);
        joinButton.gameObject.SetActive(true);
        joinLabel.gameObject.SetActive(true);

        if (networkManager == null) return;

        networkManager.Shutdown();
        HideConnectedPanel();
    }

    private void CopyRoomCodeToClipboard()
    {
        if (!string.IsNullOrEmpty(currentRoomCode))
        {
            GUIUtility.systemCopyBuffer = currentRoomCode;
            Debug.Log($"Copied room code to clipboard: {currentRoomCode}");
        }
        else 
        {
            Debug.LogWarning("No room code to copy.");
        }
    }

    private void SetPlayerName()
    {
        string enteredName = nameInput.text.Trim();

        if (!string.IsNullOrEmpty(enteredName))
        {
            playerName = enteredName;
            Debug.Log($"Player name set to: {playerName}");
        }
        else 
        {
            Debug.LogWarning("Name cannot be empty!");
        }
    }

    private void OnPlayersListChanged(NetworkListEvent<PlayerData> changeEvent)
    {
        foreach (var item in playerListItems.Values)
        {
            Destroy(item);
        }
        playerListItems.Clear();

        foreach (var player in players)
        {
            AddPlayerToList(player.clientId, player.playerName.ToString());
        }
    }

}
