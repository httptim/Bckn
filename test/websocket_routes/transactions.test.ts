/*
 * Copyright 2016 - 2024 Drew Edwards, tmpim
 *
 * This file is part of Krist.
 *
 * Krist is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Krist is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Krist. If not, see <http://www.gnu.org/licenses/>.
 *
 * For more project information, see <https://github.com/tmpim/krist>.
 */

import { expect } from "chai";

import { Address, Name, Transaction } from "../../src/database/index.js";
import { seed } from "../seed.js";
import { newConnection } from "../ws.js";

const expectTransactionExist = (id: number, to?: string, metadata?: string) => async () => {
  const tx = await Transaction.findByPk(id);
  expect(tx).to.exist;
  expect(tx).to.deep.include({ id, from: "bv8c78oh67", to: to || "bu1sdlbvwh", value: 1 });
  if (metadata) expect(tx!.op).to.equal(metadata);
};

describe("websocket routes: transactions", function() {
  before(seed);
  this.retries(4);

  async function send(data: any, privatekey?: string) {
    const ws = await newConnection(privatekey);
    expect(ws).to.nested.include({ "wsp.isOpened": true });

    const res = await ws.sendAndWait({ type: "make_transaction", ...data });
    expect(res).to.be.an("object");
    return [res, ws];
  }

  describe("make_transaction - validation", function() {
    it("should deny unauthed addresses", async function() {
      const [res, ws] = await send({ to: "bu1sdlbvwh", amount: 1, privatekey: "c" });
      expect(res).to.deep.include({ ok: false, error: "auth_failed" });
      ws.close();
    });

    it("should error with a missing 'privatekey' for guests", async function() {
      const [res, ws] = await send({});
      expect(res).to.deep.include({ ok: false, error: "missing_parameter", parameter: "privatekey" });
      ws.close();
    });

    it("should not error with a missing 'privatekey' for authed users", async function() {
      const [res, ws] = await send({}, "a");
      expect(res).to.deep.include({ ok: false, error: "missing_parameter", parameter: "to" });
      ws.close();
    });

    it("should error with a missing 'to'", async function() {
      const [res, ws] = await send({ privatekey: "a" });
      expect(res).to.deep.include({ ok: false, error: "missing_parameter", parameter: "to" });
      ws.close();
    });

    it("should error with a missing 'amount'", async function() {
      const [res, ws] = await send({ to: "bu1sdlbvwh", privatekey: "a" });
      expect(res).to.deep.include({ ok: false, error: "missing_parameter", parameter: "amount" });
      ws.close();
    });

    it("should error with an invalid 'amount'", async function() {
      const amounts = ["a", 0, -1, "0", "-1"];
      for (const amount of amounts) {
        const [res, ws] = await send({ amount, to: "bu1sdlbvwh", privatekey: "a" });
        expect(res).to.deep.include({ ok: false, parameter: "amount" });
        ws.close();
      }
    });

    it("should error with an invalid 'metadata'", async function() {
      const metadataList = ["\u0000", "\u0001", "a".repeat(256)];
      for (const metadata of metadataList) {
        const [res, ws] = await send({ amount: 1, to: "bu1sdlbvwh", privatekey: "a", metadata });
        expect(res).to.deep.include({ ok: false, error: "invalid_parameter", parameter: "metadata" });
        ws.close();
      }
    });

    it("should error with a non-existent sender", async function() {
      const [res, ws] = await send({ amount: 1, to: "bu1sdlbvwh", privatekey: "notfound" });
      expect(res).to.deep.include({ ok: false, error: "insufficient_funds" });
      ws.close();
    });

    it("should error with insufficient funds", async function() {
      const [res, ws] = await send({ amount: 11, to: "bu1sdlbvwh", privatekey: "a" });
      expect(res).to.deep.include({ ok: false, error: "insufficient_funds" });
      ws.close();
    });

    it("should error when paying to an invalid address", async function() {
      const [res, ws] = await send({ amount: 1, to: "kfartoolong", privatekey: "a" });
      expect(res).to.deep.include({ ok: false, error: "invalid_parameter", parameter: "to" });
      ws.close();
    });

    it("should error when paying to a v1 address", async function() {
      const [res, ws] = await send({ amount: 1, to: "a5dfb396d3", privatekey: "a" });
      expect(res).to.deep.include({ ok: false, error: "invalid_parameter", parameter: "to" });
      ws.close();
    });

    it("should error when paying to a name that doesn't exist", async function() {
      const [res, ws] = await send({ amount: 1, to: "notfound.bacon", privatekey: "a" });
      expect(res).to.deep.include({ ok: false, error: "name_not_found" });
      ws.close();
    });
  });

  describe("make_transaction", function() {
    it("should make a transaction", async function() {
      const [res, ws] = await send({ amount: 1, to: "bu1sdlbvwh", privatekey: "a" });
      expect(res).to.deep.include({ ok: true });
      expect(res.transaction).to.be.an("object");
      expect(res.transaction).to.deep.include({ id: 1, from: "bv8c78oh67", to: "bu1sdlbvwh", value: 1, type: "transfer" });
      expect(res.transaction.time).to.be.a("string");
      expect(res.transaction.name).to.not.be.ok;
      expect(res.transaction.metadata).to.not.be.ok;
      ws.close();
    });

    it("should exist in the database", expectTransactionExist(1));

    it("should have altered the balances", async function() {
      const from = await Address.findOne({ where: { address: "bv8c78oh67" }});
      expect(from).to.exist;
      expect(from!.balance).to.equal(9);

      const to = await Address.findOne({ where: { address: "bu1sdlbvwh" }});
      expect(to).to.exist;
      expect(to!.balance).to.equal(1);
    });

    it("should support metadata", async function() {
      const [res, ws] = await send({ amount: 1, to: "bu1sdlbvwh", privatekey: "a", metadata: "Hello, world!" });
      expect(res).to.deep.include({ ok: true });
      expect(res.transaction).to.be.an("object");
      expect(res.transaction).to.deep.include({ id: 2, from: "bv8c78oh67", to: "bu1sdlbvwh", value: 1, type: "transfer" });
      expect(res.transaction.metadata).to.equal("Hello, world!");
      ws.close();
    });

    it("should exist in the database", expectTransactionExist(2, undefined, "Hello, world!"));

    it("should create a temporary name to test", async function() {
      const name = await Name.create({ name: "test", owner: "bu1sdlbvwh", original_owner: "bu1sdlbvwh",
        registered: new Date(), unpaid: 0 });
      expect(name).to.exist;
      expect(name).to.deep.include({ name: "test", owner: "bu1sdlbvwh" });
    });

    it("should transact to a name's owner", async function() {
      const [res, ws] = await send({ amount: 1, to: "test.bacon", privatekey: "a" });
      expect(res).to.deep.include({ ok: true });
      expect(res.transaction).to.be.an("object");
      expect(res.transaction).to.deep.include({ id: 3, from: "bv8c78oh67", to: "bu1sdlbvwh", value: 1, type: "transfer" });
      expect(res.transaction.metadata).to.equal("test.bacon");
      ws.close();
    });

    it("should preserve existing metadata with a transaction to a name", async function() {
      const [res, ws] = await send({ amount: 1, to: "test.bacon", privatekey: "a", metadata: "Hello, world!" });
      expect(res).to.deep.include({ ok: true });
      expect(res.transaction).to.be.an("object");
      expect(res.transaction).to.deep.include({ id: 4, from: "bv8c78oh67", to: "bu1sdlbvwh", value: 1, type: "transfer" });
      expect(res.transaction.metadata).to.equal("test.bacon;Hello, world!");
      ws.close();
    });

    it("should support metanames", async function() {
      const [res, ws] = await send({ amount: 1, to: "meta@test.bacon", privatekey: "a" });
      expect(res).to.deep.include({ ok: true });
      expect(res.transaction).to.be.an("object");
      expect(res.transaction).to.deep.include({ id: 5, from: "bv8c78oh67", to: "bu1sdlbvwh", value: 1, type: "transfer" });
      expect(res.transaction.metadata).to.equal("meta@test.bacon");
      ws.close();
    });

    it("should support metanames and preserve metadata", async function() {
      const [res, ws] = await send({ amount: 1, to: "meta@test.bacon", privatekey: "a", metadata: "Hello, world!" });
      expect(res).to.deep.include({ ok: true });
      expect(res.transaction).to.be.an("object");
      expect(res.transaction).to.deep.include({ id: 6, from: "bv8c78oh67", to: "bu1sdlbvwh", value: 1, type: "transfer" });
      expect(res.transaction.metadata).to.equal("meta@test.bacon;Hello, world!");
      ws.close();
    });

    it("should transact to a new address", async function() {
      const [res, ws] = await send({ amount: 1, to: "bnotfound0", privatekey: "a" });
      expect(res).to.deep.include({ ok: true });
      expect(res.transaction).to.be.an("object");
      expect(res.transaction).to.deep.include({ id: 7, from: "bv8c78oh67", to: "bnotfound0", value: 1, type: "transfer" });
      ws.close();
    });

    it("should have created that address", async function() {
      const address = await Address.findOne({ where: { address: "bnotfound0" }});
      expect(address).to.exist;
      expect(address!.balance).to.equal(1);
    });

    it("should make a transaction when authed", async function() {
      const [res, ws] = await send({ amount: 1, to: "bu1sdlbvwh" }, "a");
      expect(res).to.deep.include({ ok: true });
      expect(res.transaction).to.be.an("object");
      expect(res.transaction).to.deep.include({ id: 8, from: "bv8c78oh67", to: "bu1sdlbvwh", value: 1, type: "transfer" });
      ws.close();
    });

    it("should make a transaction as another address when authed", async function() {
      const [res, ws] = await send({ amount: 1, to: "bu1sdlbvwh", privatekey: "d" }, "a");
      expect(res).to.deep.include({ ok: true });
      expect(res.transaction).to.be.an("object");
      expect(res.transaction).to.deep.include({ id: 9, from: "bw0zvuz4zn", to: "bu1sdlbvwh", value: 1, type: "transfer" });
      ws.close();
    });
  });
});
