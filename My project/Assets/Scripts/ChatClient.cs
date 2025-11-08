using UnityEngine;
using System;
using System.Net.Sockets;
using System.Text;
using System.Threading;

public class ChatClient
{
    private TcpClient client;
    private NetworkStream stream;
    private bool isConnected;
    private Thread receiveThread;

    public event Action<string> OnMessageReceived;

    public bool Connect(string host = "127.0.0.1", int port = 65432)
    {
        try
        {
            client = new TcpClient(host, port);
            stream = client.GetStream();
            isConnected = true;
            Debug.Log("Connected to Python server");

            receiveThread = new Thread(ListenForMessages);
            receiveThread.IsBackground = true;
            receiveThread.Start();

            return true;
        }
        catch (Exception e)
        {
            Debug.LogError($"Connection error: {e.Message}");
            isConnected = false;
            return false;
        }
    }

    public void SendMessage(string message)
    {
        if (!isConnected || stream == null)
        {
            Debug.LogWarning("Not connected to server, message not sent");
            return;
        }

        try
        {
            byte[] data = Encoding.UTF8.GetBytes(message + "\n");
            stream.Write(data, 0, data.Length);
            stream.Flush();
            Debug.Log($"Sent to server: {message}");
        }
        catch (Exception e)
        {
            Debug.LogError($"Error sending message: {e.Message}");
            isConnected = false;
        }
    }

    private void ListenForMessages()
    {
        try
        {
            byte[] buffer = new byte[1024];
            while (isConnected)
            {
                int bytesRead = stream.Read(buffer, 0, buffer.Length);
                if (bytesRead <= 0)
                    continue;

                string response = Encoding.UTF8.GetString(buffer, 0, bytesRead).Trim();
                Debug.Log($"[Server -> Unity] {response}");

                OnMessageReceived?.Invoke(response);
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"Error receiving data: {e.Message}");
        }
    }

    public void Disconnect()
    {
        try
        {
            isConnected = false;
            stream?.Close();
            client?.Close();
            receiveThread?.Abort();
            Debug.Log("Disconnected from server");
        }
        catch (Exception e)
        {
            Debug.LogError($"Error closing connection: {e.Message}");
        }
    }
}
