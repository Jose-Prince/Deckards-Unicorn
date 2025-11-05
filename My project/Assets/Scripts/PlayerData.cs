using Unity.Netcode;
using Unity.Collections;
using System;

public struct PlayerData : INetworkSerializable, IEquatable<PlayerData>
{
	public ulong clientId;
	public FixedString32Bytes playerName;

    public void NetworkSerialize<T>(BufferSerializer<T> serializer) where T : IReaderWriter
	{
		serializer.SerializeValue(ref clientId);
		serializer.SerializeValue(ref playerName);
	}

	public bool Equals(PlayerData other)
    {
        return clientId == other.clientId && playerName.Equals(other.playerName);
    }

    public override bool Equals(object obj)
    {
        return obj is PlayerData other && Equals(other);
    }

    public override int GetHashCode()
    {
        return HashCode.Combine(clientId, playerName);
    }
}
